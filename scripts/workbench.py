"""NeuroRaceLab Workbench v0.1: live inference and model/track switching."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pygame

from neuroracelab.agents.baseline import BaselineController
from neuroracelab.agents.neural import build_neural_controller
from neuroracelab.environment.simulator.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track
from neuroracelab.visualization.renderer import TrackRenderer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "training"
MODELS = ("baseline", "residual_rnn", "continuous_time", "equilibrium")
TRACKS = ("circuit", "oval", "procedural")


def make_controller(name: str, seed: int):
    if name == "baseline":
        return BaselineController(), "Baseline controller"
    history = RESULTS / f"{name}_seed123_history.json"
    checkpoint = RESULTS / f"{name}_seed123_checkpoint.npz"
    hidden_size = 16
    if history.exists():
        try:
            hidden_size = int(json.loads(history.read_text()).get("metadata", {}).get("hidden_size", 16))
        except (OSError, ValueError, TypeError):
            pass
    model = build_neural_controller(name, hidden_size=hidden_size, seed=seed)
    status = "random initialization (no checkpoint loaded)"
    if checkpoint.exists():
        try:
            with np.load(checkpoint) as data:
                model.load_parameters(data["parameters"])
            status = f"loaded {checkpoint.name}"
        except (OSError, KeyError, ValueError) as exc:
            status = f"checkpoint load failed: {exc}"
    return model, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, default="circuit")
    parser.add_argument("--model", choices=MODELS, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=float, default=9.0)
    parser.add_argument("--size", default="1400x900")
    parser.add_argument("--zoom", type=float, default=5.5)
    args = parser.parse_args()
    try:
        width, height = (int(v) for v in args.size.lower().split("x", 1))
        if width < 800 or height < 600:
            raise ValueError
    except ValueError:
        parser.error("--size must be WIDTHxHEIGHT, at least 800x600")

    pygame.init()
    title = "NeuroRaceLab Workbench v0.3"
    track_name, seed = args.track, args.seed
    track = build_track(track_name, seed, args.width)
    env = RaceEnvironment(track=track, seed=seed)
    obs = env.reset(seed=seed)
    model_name = args.model
    controller, model_status = make_controller(model_name, seed)
    renderer = TrackRenderer(track, size=(width, height), pixels_per_metre=args.zoom, title=title)
    font = pygame.font.Font(None, 25)
    small = pygame.font.Font(None, 21)
    panel_width = 360
    panel_gap = 18
    renderer.center[0] = max(1, (width - panel_width - panel_gap) / 2)
    clock = pygame.time.Clock()
    running, paused, ended = True, False, False
    info, total_reward = {}, 0.0
    steps, inference_ms, inference_calls = 0, 0.0, 0
    accumulator, speed_multiplier = 0.0, 1.0
    last = time.perf_counter()

    def reset_episode():
        nonlocal obs, info, total_reward, steps, ended, accumulator
        obs = env.reset(seed=seed)
        if hasattr(controller, "reset"):
            controller.reset()
        info, total_reward, steps, ended, accumulator = {}, 0.0, 0, False, 0.0

    def change_model(name: str):
        nonlocal model_name, controller, model_status
        model_name = name
        controller, model_status = make_controller(name, seed)
        reset_episode()

    def change_track(name: str):
        nonlocal track_name, track, env, obs, renderer, info
        track_name = name
        track = build_track(name, seed, args.width)
        env = RaceEnvironment(track=track, seed=seed)
        obs = env.reset(seed=seed)
        size, ppm = renderer.size, renderer.ppm
        renderer.close()
        renderer = TrackRenderer(track, size=size, pixels_per_metre=ppm, title=title)
        reset_episode()

    model_keys = {pygame.K_1: "baseline", pygame.K_2: "residual_rnn",
                  pygame.K_3: "continuous_time", pygame.K_4: "equilibrium"}
    try:
        while running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in model_keys:
                        change_model(model_keys[event.key])
                    elif event.key in (pygame.K_p, pygame.K_SPACE):
                        paused = not paused
                    elif event.key == pygame.K_BACKSPACE:
                        reset_episode()
                    elif event.key == pygame.K_t:
                        change_track(TRACKS[(TRACKS.index(track_name) + 1) % len(TRACKS)])
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        speed_multiplier = min(4.0, speed_multiplier * 1.25)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed_multiplier = max(0.25, speed_multiplier / 1.25)
                # Keep renderer navigation active: arrow keys pan the camera,
                # they do not change vehicle state or steering.
                if event.type in (pygame.QUIT, pygame.VIDEORESIZE, pygame.MOUSEWHEEL):
                    running = renderer.handle_event(event) and running
                elif event.type == pygame.KEYDOWN and event.key not in (
                    pygame.K_ESCAPE, pygame.K_q, pygame.K_p, pygame.K_SPACE,
                    pygame.K_BACKSPACE, pygame.K_t, pygame.K_EQUALS, pygame.K_PLUS,
                    pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS,
                    *model_keys.keys(),
                ):
                    running = renderer.handle_event(event) and running
                if event.type == pygame.VIDEORESIZE:
                    renderer.center[0] = max(1, (renderer.screen.get_width() - panel_width - panel_gap) / 2)

            if not paused and not ended:
                accumulator += dt * speed_multiplier
                control_dt = env.config.control_dt_s
                n = 0
                while accumulator >= control_dt and n < 8 and not ended:
                    tic = time.perf_counter()
                    action = np.asarray(controller.act(obs), dtype=np.float64)
                    inference_ms += (time.perf_counter() - tic) * 1000
                    inference_calls += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    accumulator -= control_dt
                    n += 1
                    if terminated or truncated:
                        ended, paused = True, True

            renderer.center[0] = max(1, (renderer.screen.get_width() - panel_width - panel_gap) / 2)
            renderer.draw(env.state, info=info)
            screen = renderer.screen
            screen_width, screen_height = screen.get_size()
            panel_x = screen_width - panel_width - 14
            panel_y = 14
            avg_ms = inference_ms / inference_calls if inference_calls else 0.0
            lines = [
                ("NEURORACELAB WORKBENCH v0.3", font),
                ("LIVE INFERENCE", font),
                (f"Model: {model_name}", small),
                (f"Track: {track_name} | Seed: {seed}", small),
                (f"Status: {'ENDED' if ended else 'PAUSED' if paused else 'RUNNING'}", small),
                (f"Simulation speed: {speed_multiplier:.2f}x", small),
                (f"Vehicle speed: {getattr(env.state, 'u', 0.0):.2f} m/s", small),
                (f"Steering: {getattr(env.state, 'steering', 0.0):+.3f} rad", small),
                (f"Progress: {info.get('progress', 0.0):.3f} laps", small),
                (f"Lateral error: {info.get('lateral_error', 0.0):+.2f} m", small),
                (f"Reward: {total_reward:.2f}", small),
                (f"Steps: {steps} | Sim: {info.get('elapsed_s', 0.0):.2f} s", small),
                (f"Inference: {avg_ms:.3f} ms/call", small),
                (f"End: {info.get('termination_reason') or '—'}", small),
                ("Checkpoint:", small),
                (model_status, small),
                ("MODELS", small),
                ("1 Baseline | 2 Residual", small),
                ("3 Continuous | 4 Equilibrium", small),
                ("T Track | P/Space Pause", small),
                ("Backspace Reset | +/- Speed", small),
                ("Wheel Zoom | R Reset View", small),
                ("Esc/Q Quit", small),
            ]
            # Word-wrap every line to the actual panel width. Long checkpoint
            # names/status messages get multiple rows instead of escaping the box.
            text_width = panel_width - 24
            wrapped_lines = []
            for line, fnt in lines:
                words = line.split()
                current = ""
                for word in words:
                    # Split unusually long unbroken tokens (e.g. paths/errors)
                    # as well, otherwise a single token can still overflow.
                    pieces = [word]
                    if fnt.size(word)[0] > text_width:
                        pieces = []
                        fragment = ""
                        for char in word:
                            if fragment and fnt.size(fragment + char)[0] > text_width:
                                pieces.append(fragment)
                                fragment = char
                            else:
                                fragment += char
                        if fragment:
                            pieces.append(fragment)
                    for piece in pieces:
                        candidate = piece if not current else f"{current} {piece}"
                        if current and fnt.size(candidate)[0] > text_width:
                            wrapped_lines.append((current, fnt))
                            current = piece
                        else:
                            current = candidate
                wrapped_lines.append((current, fnt))

            line_gap = 5
            rendered_rows = []
            for line, fnt in wrapped_lines:
                surface = fnt.render(line, True, (238, 242, 247))
                rendered_rows.append((surface, 4 if fnt is small else 7))
            content_height = sum(surface.get_height() + gap for surface, gap in rendered_rows)
            panel_height = min(screen_height - 28, content_height + 24)
            panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            panel.fill((15, 22, 31, 238))
            screen.blit(panel, (panel_x, panel_y))
            y = panel_y + 12
            for surface, gap in rendered_rows:
                if y + surface.get_height() > panel_y + panel_height - 8:
                    break
                screen.blit(surface, (panel_x + 12, y))
                y += surface.get_height() + gap
            pygame.display.flip()
            clock.tick(60)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
