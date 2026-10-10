"""Reel intro in the 3Blue1Brown style: the title is written, slides up as the arm photo fades in inside a drawn
frame, then the ten pipeline steps are drawn left to right, each joined to the next by a growing arrow."""
from manim import *

config.background_color = "#0E0E0E"
REAL, SIM, COMP = YELLOW_D, BLUE_C, GREY_B
STEPS = [("01", "Phone", "capture", REAL), ("02", "Gaussian", "splat", SIM), ("03", "Real-to-sim", "alignment", SIM),
         ("04", "Sim", "demos", SIM), ("05", "Recovery", "data", SIM), ("06", "Data", "audit", COMP),
         ("07", "Real", "demos", REAL), ("08", "Train \\&", "iterate", COMP), ("09", "Sim", "eval", SIM),
         ("10", "Real arm", "deploy", REAL)]


class Intro(Scene):
    def construct(self):
        # 1. title, written large in the middle
        words = VGroup(Tex(r"\textbf{Real}", color=REAL), Tex(r"\textbf{Sim}", color=SIM), Tex(r"\textbf{Real}", color=REAL))
        for w in words:
            w.scale(2.4)
        words.arrange(RIGHT, buff=1.3).move_to(UP * 0.4)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.2, stroke_width=6, color=GREY_A,
                                max_tip_length_to_length_ratio=0.3) for a, b in zip(words, words[1:])])
        self.play(LaggedStart(Write(words[0]), GrowArrow(arrows[0]), Write(words[1]), GrowArrow(arrows[1]),
                              Write(words[2]), lag_ratio=0.45), run_time=1.7)

        # 2. title moves up and right; the photo fades in inside a frame that is drawn around it
        photo = ImageMobject("arm.png").set_height(4.1).move_to([-4.6, 1.1, 0])
        frame = SurroundingRectangle(photo, buff=0.06, color=GREY_B, stroke_width=2.5, corner_radius=0.12)
        title = VGroup(words, arrows)
        target = title.copy().scale(0.62)
        target.next_to(photo, RIGHT, buff=0.75).align_to(photo, UP).shift(DOWN * 0.15)
        self.play(Transform(title, target), FadeIn(photo, scale=0.96), Create(frame), run_time=0.9)

        # 3. subtitle and the numbers line
        sub = VGroup(Tex("Teaching an SO-101 arm"), Tex("to put a cube in a cup")).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sub.scale(1.05).next_to(title, DOWN, buff=0.55).align_to(title, LEFT)
        tag = Tex("40 real + 4,013 sim demos", color=GREY_B).scale(0.72)
        tag.next_to(sub, DOWN, buff=0.4).align_to(sub, LEFT)
        self.play(Write(sub), run_time=0.9)
        self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.4)

        # 4. the ten steps, drawn left to right and joined by arrows
        n, gap, y = len(STEPS), 0.3, -2.45
        w = (13.7 - (n - 1) * gap) / n
        boxes, labels, nums, links = [], [], [], []
        for i, (num, l1, l2, col) in enumerate(STEPS):
            x = -6.85 + w / 2 + i * (w + gap)
            box = RoundedRectangle(corner_radius=0.1, width=w, height=0.92, stroke_color=col, stroke_width=2.5)
            box.set_fill(col, opacity=0.16).move_to([x, y, 0])
            lab = Tex(r"\begin{tabular}{c}" + l1 + r"\\" + l2 + r"\end{tabular}", color=WHITE).scale(0.5)
            if lab.width > w - 0.12:
                lab.scale_to_fit_width(w - 0.12)
            lab.move_to(box)
            nb = Tex(num, color=col).scale(0.5).next_to(box, UP, buff=0.1)
            boxes.append(box); labels.append(lab); nums.append(nb)
        for a, b in zip(boxes, boxes[1:]):
            links.append(Arrow(a.get_right(), b.get_left(), buff=0.03, stroke_width=3, color=GREY_B,
                               max_tip_length_to_length_ratio=0.5, max_stroke_width_to_length_ratio=12))
        seq = []
        for i in range(n):
            seq.append(AnimationGroup(DrawBorderThenFill(boxes[i]), Write(labels[i]), FadeIn(nums[i]), run_time=0.6))
            if i < n - 1:
                seq.append(GrowArrow(links[i], run_time=0.25))
        self.play(LaggedStart(*seq, lag_ratio=0.42), run_time=3.8)

        # 5. legend, a nod to the last step, then hold
        legend = Tex("real", r"\qquad{}", "sim", r"\qquad{}", "compute").scale(0.62).move_to([0, -3.5, 0])  # one Tex: one baseline
        for k, c in ((0, REAL), (2, SIM), (4, COMP)):
            legend[k].set_color(c)
        self.play(Write(legend), run_time=0.5)
        self.play(Circumscribe(VGroup(boxes[-1], nums[-1]), color=REAL, buff=0.08), run_time=0.7)
        self.wait(1.0)
