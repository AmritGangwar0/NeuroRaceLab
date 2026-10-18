"""Pygame population-training preview for NeuroRaceLab.

Keys: EA/ES switch (E), pause (Space), next generation (N), reset (R), quit (Esc).
All cars are neural controllers. There is no keyboard steering.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pygame

from neuroracelab.agents.neural import build_neural_controller
from neuroracelab.environment.simulator.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track
from neuroracelab.training.population import PopulationConfig, PopulationTrainer
from neuroracelab.visualization.renderer import TrackRenderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("oval", "circuit", "procedural"), default="oval")
    parser.add_argument("--architecture", choices=("residual_rnn", "continuous_time", "equilibrium"), default="residual_rnn")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=float, default=9.0)
    args = parser.parse_args()

    track = build_track(args.track, args.seed, args.width)
    template = build_neural_controller(args.architecture, hidden_size=args.hidden_size, seed=args.seed)
    config = PopulationConfig(size=args.population, seed=args.seed)
    trainer = PopulationTrainer(template.parameter_count, config)
    envs = [RaceEnvironment(track=track, seed=args.seed + i) for i in range(args.population)]
    observations = [env.reset(seed=args.seed + i) for i, env in enumerate(envs)]
    controllers = [build_neural_controller(args.architecture, hidden_size=args.hidden_size, seed=args.seed + i) for i in range(args.population)]
    for c, candidate in zip(controllers, trainer.population):
        c.load_parameters(candidate.parameters)

    renderer = TrackRenderer(track, size=(1440, 900), pixels_per_metre=5.0, title="NeuroRaceLab | Population Evolution Preview")
    font = pygame.font.Font(None, 24)
    small = pygame.font.Font(None, 20)
    paused = False
    generation_done = [False] * args.population
    fitness_sum = np.zeros(args.population, dtype=np.float64)
    step_count = np.zeros(args.population, dtype=np.int64)
    collision_count = 0
    speed = 1
    running = True
    last = time.perf_counter()
    report_dir = Path("results/population")
    report_dir.mkdir(parents=True, exist_ok=True)
    palette = [(66, 180, 255), (255, 190, 70), (244, 94, 94), (160, 130, 255), (80, 220, 160), (255, 120, 210)]

    def save_report() -> Path:
        """Write generation-level JSON and CSV reports."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        payload = {
            "track": args.track, "seed": args.seed, "architecture": args.architecture,
            "hidden_size": args.hidden_size, "population_size": args.population,
            "final_generation": trainer.generation, "current_mode": trainer.mode,
            "generation_history": trainer.history,
            "current_population": [
                {"candidate": i, "fitness": float(c.fitness), "progress": float(c.progress),
                 "crashed": bool(c.crashed)}
                for i, c in enumerate(trainer.population)
            ],
        }
        json_path = report_dir / f"population_{stamp}.json"
        csv_path = report_dir / f"population_{stamp}.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["generation", "mode", "best_fitness", "mean_fitness", "best_progress", "crash_rate"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trainer.history)
        return json_path

    def reset_generation() -> None:
        nonlocal observations, generation_done, fitness_sum, step_count, collision_count
        observations = [env.reset(seed=args.seed + trainer.generation * 1000 + i) for i, env in enumerate(envs)]
        generation_done = [False] * args.population
        fitness_sum = np.zeros(args.population, dtype=np.float64)
        step_count = np.zeros(args.population, dtype=np.int64)
        collision_count = 0
        for c, candidate in zip(controllers, trainer.population):
            c.reset()
            c.load_parameters(candidate.parameters)

    def finish_generation() -> None:
        for i, candidate in enumerate(trainer.population):
            candidate.fitness = float(fitness_sum[i])
            candidate.progress = float(envs[i].progress)
            candidate.crashed = bool(getattr(envs[i], "_population_crashed", False))
        trainer.evolve()
        reset_generation()

    try:
        while running:
            dt = min(time.perf_counter() - last, 0.1)
            last = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_e:
                        trainer.switch_mode()
                    elif event.key == pygame.K_n:
                        finish_generation()
                    elif event.key == pygame.K_r:
                        reset_generation()
                    elif event.key == pygame.K_g:
                        path = save_report()
                        print(f"Population report saved: {path}")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        speed = min(8, speed + 1)
                    elif event.key == pygame.K_MINUS:
                        speed = max(1, speed - 1)
                # Arrow keys are intentionally not forwarded: renderer's legacy
                # handler uses them to pan the camera, which looked like car control.
                if not (event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT
                )):
                    renderer.handle_event(event)

            if not paused:
                for _ in range(speed):
                    all_done = True
                    for i, (env, controller) in enumerate(zip(envs, controllers)):
                        if generation_done[i]:
                            continue
                        all_done = False
                        action = controller.act(observations[i])
                        obs, reward, terminated, truncated, info = env.step(action)
                        observations[i] = obs
                        step_count[i] += 1
                        fitness_sum[i] += reward
                        # Hard collision: the car's half-length is conservatively
                        # approximated as 1.2 m along the track normal.
                        half_width = float(track.width_at_s(info.get("track_s", 0.0))) / 2.0
                        if abs(float(info.get("lateral_error", 0.0))) + 1.2 >= half_width:
                            terminated = True
                            info["termination_reason"] = "collision_green_area"
                            env._population_crashed = True
                        if terminated or truncated:
                            generation_done[i] = True
                            candidate = trainer.population[i]
                            candidate.fitness = float(fitness_sum[i])
                            candidate.progress = float(info.get("progress", 0.0))
                            candidate.crashed = bool(info.get("termination_reason") != "lap_complete")
                            if candidate.crashed:
                                fitness_sum[i] -= 25.0
                    if all(generation_done):
                        finish_generation()
                        break

            # TrackRenderer draws track/background and first car; overlay the rest.
            renderer.draw(envs[0].state, info={"progress": envs[0].progress})
            for i, env in enumerate(envs):
                state = env.state
                angle = -state.psi
                length_px = max(7, int(2.8 * renderer.ppm * renderer.zoom))
                width_px = max(4, int(1.4 * renderer.ppm * renderer.zoom))
                p = renderer._screen_point(np.array([state.x, state.y]))
                ca, sa = math.cos(angle), math.sin(angle)
                corners = []
                for lx, ly in ((length_px / 2, width_px / 2), (length_px / 2, -width_px / 2), (-length_px / 2, -width_px / 2), (-length_px / 2, width_px / 2)):
                    corners.append((int(p[0] + lx * ca - ly * sa), int(p[1] + lx * sa + ly * ca)))
                color = palette[i % len(palette)]
                if generation_done[i]:
                    color = (105, 105, 105)
                pygame.draw.polygon(renderer.screen, color, corners)
                pygame.draw.polygon(renderer.screen, (20, 24, 30), corners, 1)
                label = small.render(str(i), True, (255, 255, 255))
                renderer.screen.blit(label, (p[0] + 4, p[1] + 2))

            # HUD uses measured pixel widths and wrapped lines; no fixed-height overflow.
            best = max(trainer.population, key=lambda c: c.fitness)
            active_count = sum(not x for x in generation_done)
            lines = [
                "NEURORACELAB | POPULATION WORKBENCH",
                f"Algorithm: {trainer.mode}    Generation: {trainer.generation}",
                f"Architecture: {args.architecture}    Hidden size: {args.hidden_size}",
                f"Population: {args.population}    Active cars: {active_count}",
                f"Best fitness: {best.fitness:.3f}    Progress: {best.progress:.3f}",
                "Green-area contact: terminate candidate + fitness penalty",
                "E: switch EA/ES    N: next generation    Space: pause",
                "R: reset generation    +/-: simulation speed    G: save report",
                "Esc: quit    Neural-network control only; no keyboard steering",
            ]
            max_text_width = min(760, renderer.screen.get_width() - 32)
            wrapped = []
            for line in lines:
                words = line.split()
                current = ""
                for word in words:
                    trial = f"{current} {word}".strip()
                    if current and small.size(trial)[0] > max_text_width - 24:
                        wrapped.append(current)
                        current = word
                    else:
                        current = trial
                if current:
                    wrapped.append(current)
            line_height = small.get_linesize() + 3
            panel_width = min(max_text_width, max(small.size(line)[0] for line in wrapped) + 24)
            panel_height = len(wrapped) * line_height + 18
            panel = pygame.Rect(12, 12, panel_width, panel_height)
            pygame.draw.rect(renderer.screen, (20, 25, 34), panel, border_radius=8)
            pygame.draw.rect(renderer.screen, (90, 100, 115), panel, 1, border_radius=8)
            y = panel.y + 9
            for line in wrapped:
                surface = small.render(line, True, (235, 239, 245))
                renderer.screen.blit(surface, (panel.x + 12, y))
                y += line_height
            pygame.display.flip()
            renderer.clock.tick(60)
    finally:
        if trainer.history:
            path = save_report()
            print(f"Final population report saved: {path}")
        renderer.close()


if __name__ == "__main__":
    main()
