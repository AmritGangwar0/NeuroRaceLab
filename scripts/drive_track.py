from __future__ import annotations

import argparse
import math

import pygame

from neuroracelab.environment.simulator.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track
from neuroracelab.visualization.controls import keys_to_action
from neuroracelab.visualization.renderer import TrackRenderer


def draw_sensor_rays(renderer: TrackRenderer, env: RaceEnvironment, observation) -> None:
    """Draw range-sensor observations over the rendered track."""
    state = env.state
    origin = (state.x, state.y)
    for relative_angle, normalized_range in zip(env.sensor.angles, observation[:env.sensor.ray_count]):
        angle = state.psi + float(relative_angle)
        distance = float(normalized_range) * env.sensor.max_range_m
        endpoint = (origin[0] + math.cos(angle) * distance,
                    origin[1] + math.sin(angle) * distance)
        pygame.draw.line(renderer.screen, (255, 174, 66),
                         renderer._screen_point(origin), renderer._screen_point(endpoint), 1)
    pygame.display.flip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the NeuroRaceLab car manually.")
    parser.add_argument("--track", choices=("circuit", "oval", "procedural"), default="circuit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=float, default=9.0)
    parser.add_argument("--zoom", type=float, default=5.5)
    parser.add_argument("--size", default="1200x800")
    args = parser.parse_args()

    try:
        width, height = (int(v) for v in args.size.lower().split("x", 1))
        if width <= 0 or height <= 0:
            raise ValueError
    except ValueError:
        parser.error("--size must look like 1200x800 with positive dimensions")

    track = build_track(args.track, args.seed, args.width)
    env = RaceEnvironment(track=track, seed=args.seed)
    observation = env.reset(seed=args.seed)
    renderer = TrackRenderer(track, size=(width, height), pixels_per_metre=args.zoom)
    print(f"Track: {track.track_id} | seed: {args.seed}")
    print("Controls: Up accelerate, Down brake, Left/Right steer, Space reset, R reset camera, Esc quit.")

    running = True
    accumulator = 0.0
    last_ticks = pygame.time.get_ticks()
    info = {}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                observation = env.reset(seed=args.seed)
                accumulator = 0.0
                info = {}
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                renderer.zoom = 1.0
                renderer.pan[:] = 0.0
                renderer._background = None
            elif event.type in (pygame.VIDEORESIZE, pygame.MOUSEWHEEL):
                renderer.handle_event(event)

        keys = pygame.key.get_pressed()
        steering, longitudinal = keys_to_action(
            left=keys[pygame.K_LEFT], right=keys[pygame.K_RIGHT],
            throttle=keys[pygame.K_UP], brake=keys[pygame.K_DOWN],
        )
        now = pygame.time.get_ticks()
        accumulator += min((now - last_ticks) / 1000.0, 0.1)
        last_ticks = now
        step_dt = env.config.control_dt_s
        while accumulator >= step_dt:
            observation, _, terminated, truncated, info = env.step((steering, longitudinal))
            accumulator -= step_dt
            if terminated or truncated:
                print(f"Episode ended: {info.get('termination_reason')} | progress={info.get('progress', 0.0):.3f}")
                observation = env.reset(seed=args.seed)
                accumulator = 0.0
                break

        renderer.draw(env.state, info=info)
        draw_sensor_rays(renderer, env, observation)
        renderer.clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
