"""Reel closing card in the 3Blue1Brown style, stats from most to least important: real-arm success of real + sim vs
real-only (bars that grow while the counts tick up, then the gap), then why it works and the supporting numbers."""
from manim import *

config.background_color = "#0E0E0E"
REAL, SIM = YELLOW_D, BLUE_C
RUNS = 21
BAR_X0, BAR_W, BAR_H = -3.85, 4.4, 0.56
COUNT_W = 0.5  # room for a two-digit count left of "/ 21"


def bar_row(label, wins, y, color):
    lab = label.scale(0.95)
    lab.move_to([BAR_X0 - 0.3, y, 0], aligned_edge=RIGHT)
    outline = Rectangle(width=BAR_W, height=BAR_H, stroke_color=GREY_D, stroke_width=2)
    outline.move_to([BAR_X0 + BAR_W / 2, y, 0])
    fill = Rectangle(width=BAR_W * wins / RUNS, height=BAR_H, stroke_width=0).set_fill(color, 0.9)
    fill.move_to([BAR_X0, y, 0], aligned_edge=LEFT)
    count = Integer(0).scale(1.05)
    tail = Tex(rf"/\,{RUNS}", r"\ \ ({:.0f}\%)".format(100 * wins / RUNS)).scale(1.05)
    tail[1].set_color(GREY_B).scale(0.75, about_edge=LEFT)
    tail.move_to([BAR_X0 + BAR_W + 0.3 + COUNT_W + 0.12, y, 0], aligned_edge=LEFT)
    # keep the ticking count glued to "/ 21", on the same baseline
    count.add_updater(lambda m: m.next_to(tail, LEFT, buff=0.12).align_to(tail[0][-1], DOWN))
    return lab, outline, fill, count, tail, wins


class Outro(Scene):
    def construct(self):
        # 1. the headline comparison
        head = Tex(r"\textbf{Success on the real arm}", r"\ \ by training data").scale(0.95)
        head[1].set_color(GREY_B)
        head.move_to([-6.45, 3.05, 0], aligned_edge=LEFT)
        rows = [bar_row(Tex("Real", " + ", "sim").set_color_by_tex("Real", REAL).set_color_by_tex("sim", SIM), 19, 2.05,
                        SIM),
                bar_row(Tex("Real only", color=REAL), 6, 1.1, REAL)]
        note = Tex(r"real only: same 40 real demos, 40k + 80k-step checkpoints (4\,/\,16 + 2\,/\,5)",
                   color=GREY_B).scale(0.48)
        note.move_to([BAR_X0, 0.6, 0], aligned_edge=LEFT)
        self.play(Write(head), *[Create(r[1]) for r in rows], *[FadeIn(r[0], shift=RIGHT * 0.2) for r in rows], run_time=0.6)
        self.play(*[GrowFromEdge(r[2], LEFT) for r in rows], *[FadeIn(r[4][0]) for r in rows],
                  *[ChangeDecimalToValue(r[3], r[5]) for r in rows], run_time=1.1, rate_func=smooth)
        for r in rows:
            r[3].clear_updaters()
        brace = Brace(VGroup(rows[0][4], rows[1][4], rows[0][3], rows[1][3]), RIGHT, buff=0.15, color=GREY_B)
        gain = VGroup(Tex(r"\textbf{3}$\times$", color=WHITE).scale(1.1),
                      Tex("the success rate", color=GREY_B).scale(0.65)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        gain.next_to(brace, RIGHT, buff=0.15)
        self.play(*[FadeIn(r[4][1]) for r in rows], GrowFromCenter(brace), Write(gain), FadeIn(note), run_time=0.5)
        self.play(Circumscribe(VGroup(rows[0][3], rows[0][4]), color=SIM, buff=0.1), run_time=0.5)

        # 2. the supporting numbers, most important first
        rule = Line([-6.45, 0.3, 0], [6.45, 0.3, 0], stroke_width=1.5, color=GREY_D)
        stats = [
            (Tex(r"\textbf{336}", " vs ", r"\textbf{4}"), (SIM, GREY_B, REAL),
             "sim vs real demos within 5 cm of where a real-only run failed"),
            (Tex(r"\textbf{88\%}"), (SIM,), "sim success on 50 held-out randomized scenes"),
            (Tex(r"\textbf{45 s}"), (REAL,), "of phone video, rebuilt as the Gaussian-splat room in the sim"),
            (Tex(r"\textbf{40 ms}"), (WHITE,), "policy inference on a Jetson Orin Nano, next to the arm"),
        ]
        lines = []
        for k, (num, cols, text) in enumerate(stats):
            for part, c in zip(num, cols):
                part.set_color(c)
            y = -0.4 - 0.88 * k
            num.scale(1.05).move_to([-6.45, y, 0], aligned_edge=LEFT)
            desc = Tex(text, color=GREY_A).scale(0.74).move_to([BAR_X0, y, 0], aligned_edge=LEFT)
            lines.append(VGroup(num, desc))
        self.play(Create(rule), run_time=0.3)
        self.play(LaggedStart(*[AnimationGroup(Write(l[0]), FadeIn(l[1], shift=RIGHT * 0.15)) for l in lines],
                              lag_ratio=0.55), run_time=2.0)
        self.wait(2.8)
