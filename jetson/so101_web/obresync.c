// LD_PRELOAD shim for the Orbbec OpenNI2 driver: keeps the Astra's PS1080 command/reply stream in sync.
//
// Measured device behaviour: each control IN returns the OLDEST queued reply (the queue lags several commands
// behind and the lag never shrinks), a read only returns data after a new command, replies queued across an
// aborted read can lose their first 4 (rarely 8) bytes, and a zero-length read leaves a reply queued (lag +1).
// So on each driver read we pump harmless GetVersion "filler" commands, one read per filler, until the reply with
// the driver's (opcode, id) comes out, repairing a header that lost its magic+size. Limits scale with the measured
// lag. The shim NEVER resends a driver command: a version that resent commands hung the sensor firmware.
// Set OBRESYNC_LOG=<file> to trace every command/reply. Build: gcc -shared -fPIC -O2 -o obresync.so obresync.c -ldl -lpthread
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct libusb_device_handle libusb_device_handle;
typedef int (*ctrl_fn)(libusb_device_handle *, uint8_t, uint8_t, uint16_t, uint16_t, unsigned char *, uint16_t, unsigned int);

#define CMD_MAGIC 0x4d47
#define RPL_MAGIC 0x4252
#define READ_TMO 1000
#define HARD_CAP 4096        // max reads per driver command / flush (the lag grows ~1-2 per process start)
#define MAX_TIMEOUTS 3       // consecutive NAK timeouts after a filler: the device stopped answering
#define MAX_ZLPS 20          // consecutive zero-length reads
#define LAG_WARN 300         // above this an open takes ~15 s: ask for a replug

static ctrl_fn real_ctrl;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int verbose = -1;
static int have_pending, flushed, lag_est = -1;
static uint16_t pending_id, pending_op, pending_len;
static unsigned char pending_cmd[512];
static uint16_t filler_seq, filler_base;
static unsigned long n_pumps, n_repairs;
static FILE *trace;

static void *libusb(void) {
    static void *h;
    if (!h) h = dlopen("libusb-1.0.so.0", RTLD_NOW | RTLD_GLOBAL);
    return h;
}

static void init(void) {
    if (!real_ctrl) real_ctrl = (ctrl_fn)dlsym(libusb(), "libusb_control_transfer");
    if (verbose < 0) {
        const char *v = getenv("OBRESYNC_VERBOSE"), *path = getenv("OBRESYNC_LOG");
        verbose = v ? atoi(v) : 0;
        if (path && (trace = fopen(path, "a"))) { setvbuf(trace, NULL, _IOLBF, 0); verbose = 1; }
    }
}
#include <stdarg.h>
#include <time.h>
static void LOG(const char *fmt, ...) {
    if (!verbose) return;
    FILE *f = trace ? trace : stderr;
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    fprintf(f, "%ld.%03ld [%d] ", (long)t.tv_sec, t.tv_nsec / 1000000, (int)getpid());
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap);
}

static uint16_t u16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// Filler ids: 0x8000 | (pid & 7) << 12 | 12-bit seq. Driver ids are small, so they never collide with fillers.
static int send_filler(libusb_device_handle *h, uint16_t *id_out) {
    unsigned char pkt[8] = {0x47, 0x4d, 0, 0, 0, 0, 0, 0};
    if (!filler_base) filler_base = (uint16_t)(0x8000 | ((getpid() & 0x7) << 12));
    uint16_t id = filler_base | (filler_seq++ & 0xfff);
    pkt[6] = id & 0xff; pkt[7] = id >> 8;
    if (id_out) *id_out = id;
    n_pumps++;
    return real_ctrl(h, 0x40, 0, 0, 0, pkt, 8, READ_TMO);
}

// Reply id: full header at +6, header that lost magic+size at +2.
static int reply_id(const unsigned char *buf, int r, uint16_t *id) {
    if (r >= 8 && u16(buf) == RPL_MAGIC) { *id = u16(buf + 6); return 1; }
    if (r >= 4 && u16(buf) != RPL_MAGIC) { *id = u16(buf + 2); return 1; }
    return 0;
}

// Drain replies left by earlier processes: pump until our own first filler's reply comes back. Its position = lag.
static void flush_stale(libusb_device_handle *h) {
    uint16_t first, id;
    unsigned char buf[512];
    int timeouts = 0;
    if (send_filler(h, &first) != 8) { LOG("obresync: flush: filler send failed (sensor not answering)\n"); return; }
    for (int i = 0; i < HARD_CAP; i++) {
        int r = real_ctrl(h, 0xC0, 0, 0, 0, buf, sizeof(buf), READ_TMO);
        if (r == -7 && ++timeouts >= MAX_TIMEOUTS) break;
        if (r > 0) timeouts = 0;
        if (r > 0 && reply_id(buf, r, &id) && id == first) {
            lag_est = i;
            fprintf(stderr, "obresync: lag %d\n", lag_est);
            LOG("obresync: lag %d\n", lag_est);
            if (lag_est > LAG_WARN)
                fprintf(stderr, "obresync: sensor reply queue lag is high (%d); unplug and replug the Astra soon\n", lag_est);
            return;
        }
        if (send_filler(h, NULL) != 8) return;
    }
    fprintf(stderr, "obresync: could not measure lag (flush gave up)\n");
}

// Copy the reply for the pending command into out; returns its length, or 0 if buf is some other reply.
static int classify(const unsigned char *buf, int r, unsigned char *out, int outlen) {
    if (r >= 10 && u16(buf) == RPL_MAGIC) {
        if (u16(buf + 6) == pending_id && u16(buf + 4) == pending_op) { int n = r < outlen ? r : outlen; memcpy(out, buf, n); return n; }
        return 0;
    }
    if (r >= 6 && r + 4 <= 512 && u16(buf) == pending_op && u16(buf + 2) == pending_id && r + 4 <= outlen) {
        uint16_t words = (uint16_t)((r + 4 - 8) / 2);
        out[0] = RPL_MAGIC & 0xff; out[1] = RPL_MAGIC >> 8; out[2] = words & 0xff; out[3] = words >> 8;
        memcpy(out + 4, buf, r);
        n_repairs++;
        return r + 4;
    }
    return 0;
}

int libusb_control_transfer(libusb_device_handle *h, uint8_t rt, uint8_t req, uint16_t val, uint16_t idx,
                            unsigned char *data, uint16_t len, unsigned int timeout) {
    init();
    int vendor_cmd = (rt == 0x40 && req == 0 && len >= 8 && data && u16(data) == CMD_MAGIC);
    int vendor_rpl = (rt == 0xC0 && req == 0);
    if (!vendor_cmd && !vendor_rpl) return real_ctrl(h, rt, req, val, idx, data, len, timeout);

    pthread_mutex_lock(&mu);
    int ret;
    if (vendor_cmd) {
        if (!flushed) { flushed = 1; flush_stale(h); }
        ret = real_ctrl(h, rt, req, val, idx, data, len, timeout);
        if (ret == len) {
            have_pending = 1; pending_op = u16(data + 4); pending_id = u16(data + 6);
            pending_len = len < sizeof(pending_cmd) ? len : sizeof(pending_cmd);
            memcpy(pending_cmd, data, pending_len);
            LOG("obresync: cmd op=%#x id=%#x len=%u\n", pending_op, pending_id, len);
        }
    } else if (!have_pending) {
        ret = real_ctrl(h, rt, req, val, idx, data, len, timeout);   // nothing outstanding: plain read
    } else {
        unsigned char buf[512];
        int found = 0, timeouts = 0, zlps = 0;
        // Unknown lag (flush failed): search the whole cap and never resend, since nothing says the reply was lost.
        int limit = lag_est >= 0 ? 2 * lag_est + 64 : HARD_CAP;
        if (limit > HARD_CAP) limit = HARD_CAP;
        ret = -7;  // LIBUSB_ERROR_TIMEOUT unless found
        for (int i = 0; i <= limit && !found; i++) {
            if (i > 0 && send_filler(h, NULL) != 8) { LOG("obresync: filler send failed\n"); break; }
            int r = real_ctrl(h, 0xC0, 0, 0, 0, buf, sizeof(buf), READ_TMO);
            if (r < 0 && r != -7) { ret = r; break; }                  // real USB error: pass it up
            if (r == -7) { if (++timeouts >= MAX_TIMEOUTS) break; continue; }
            timeouts = 0;
            if (r == 0) { if (++zlps >= MAX_ZLPS) break; continue; }
            zlps = 0;
            int n = classify(buf, r, data, len);
            if (n > 0) { ret = n; found = 1; lag_est = i; break; }
            uint16_t rid;
            if (reply_id(buf, r, &rid)) LOG("obresync:   skip reply id=%#x len=%d\n", rid, r);
        }
        have_pending = 0;
        LOG("obresync: id=%#x -> %d (pumps %lu, repairs %lu, lag %d)\n", pending_id, ret, n_pumps, n_repairs, lag_est);
    }
    pthread_mutex_unlock(&mu);
    return ret;
}
