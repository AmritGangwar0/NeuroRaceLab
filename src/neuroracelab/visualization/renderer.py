from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised only without render extra
    raise ImportError(
        "Track rendering requires the optional dependency. "
        "Install it with: python -m pip install -e '.[render]'"
    ) from exc

from neuroracelab.environment.tracks.track import Track


class TrackRenderer:
    """Pygame renderer for Track geometry; it does not alter simulation state."""

    GRASS = (65, 116, 66)
    GRASS_DARK = (57, 103, 59)
    ASPHALT = (72, 75, 79)
    EDGE = (224, 224, 216)
    WHITE = (242, 242, 234)
    RED = (194, 48, 47)
    CAR = (45, 145, 238)
    WINDOW = (24, 38, 52)

    def __init__(
        self,
        track: Track,
        size: tuple[int, int] = (1200, 800),
        pixels_per_metre: float = 5.5,
        title: str = "NeuroRaceLab | Track Viewer",
    ) -> None:
        if pixels_per_metre <= 0:
            raise ValueError("pixels_per_metre must be positive")
        pygame.init()
        self.track = track
        self.size = size
        self.ppm = float(pixels_per_metre)
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.center = np.array([size[0] / 2.0, size[1] / 2.0], dtype=np.float64)
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=np.float64)
        self._background: pygame.Surface | None = None
        self._background_size: tuple[int, int] | None = None

    def close(self) -> None:
        pygame.quit()

    def _screen_point(self, point: np.ndarray) -> tuple[int, int]:
        p = np.asarray(point, dtype=np.float64)
        scale = self.ppm * self.zoom
        x = self.center[0] + self.pan[0] + p[0] * scale
        y = self.center[1] + self.pan[1] - p[1] * scale
        return int(round(x)), int(round(y))

    def _polygon(self, points: np.ndarray) -> list[tuple[int, int]]:
        return [self._screen_point(p) for p in points]

    def _build_background(self) -> None:
        width, height = self.screen.get_size()
        bg = pygame.Surface((width, height))
        bg.fill(self.GRASS)

        # Subtle, deterministic grass texture, generated once per window size.
        spacing = 38
        for y in range(12, height, spacing):
            offset = (y // spacing % 2) * 18
            for x in range(12 + offset, width, spacing):
                pygame.draw.circle(bg, self.GRASS_DARK, (x, y), 1)

        left, right = self.track.boundaries()
        road_polygon = self._polygon(np.vstack((left, right[::-1])))
        pygame.draw.polygon(bg, self.EDGE, road_polygon)
        pygame.draw.lines(bg, self.EDGE, True, self._polygon(left), max(2, int(0.35 * self.ppm)))
        pygame.draw.lines(bg, self.EDGE, True, self._polygon(right), max(2, int(0.35 * self.ppm)))

        # Centerline dashes are a visual cue only, not a physics feature.
        pts = self.track.centerline
        dash_stride = max(1, len(pts) // 90)
        for i in range(0, len(pts), dash_stride * 2):
            j = min(i + dash_stride, len(pts) - 1)
            pygame.draw.line(
                bg,
                (188, 190, 188),
                self._screen_point(pts[i]),
                self._screen_point(pts[j]),
                max(1, int(0.12 * self.ppm)),
            )

        # Start/finish line perpendicular to the local tangent.
        start = self.track.centerline[0]
        tangent = self.track.tangent_at_index(0)
        normal = np.array([-tangent[1], tangent[0]])
        half = float(self.track.width_at_s(0.0)) * 0.47
        a, b = start - normal * half, start + normal * half
        pygame.draw.line(bg, self.WHITE, self._screen_point(a), self._screen_point(b), max(3, int(0.6 * self.ppm)))
        self._background = bg
        self._background_size = (width, height)

    def _draw_car(self, state: Any) -> None:
        x = float(state.x)
        y = float(state.y)
        heading = float(state.psi)
        # Vehicle footprint is intentionally schematic, not a dynamics model.
        length, width = 4.2, 1.85
        forward = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-forward[1], forward[0]])
        center = np.array([x, y])
        corners = [
            center + forward * length / 2 + side * width / 2,
            center + forward * length / 2 - side * width / 2,
            center - forward * length / 2 - side * width / 2,
            center - forward * length / 2 + side * width / 2,
        ]
        polygon = self._polygon(np.asarray(corners))
        pygame.draw.polygon(self.screen, self.CAR, polygon)
        pygame.draw.polygon(self.screen, self.WINDOW, polygon, max(1, int(0.12 * self.ppm)))
        cabin = center + forward * 0.25
        cabin_corners = [
            cabin + forward * 0.65 + side * width * 0.32,
            cabin + forward * 0.65 - side * width * 0.32,
            cabin - forward * 0.65 - side * width * 0.32,
            cabin - forward * 0.65 + side * width * 0.32,
        ]
        pygame.draw.polygon(self.screen, self.WINDOW, self._polygon(np.asarray(cabin_corners)))

    def draw(self, state: Any | None = None, *, info: dict | None = None) -> None:
        if self._background is None or self._background_size != self.screen.get_size():
            self._build_background()
        self.screen.blit(self._background, (0, 0))
        if state is not None:
            self._draw_car(state)

        help_text = "Wheel/trackpad: zoom | Arrows: pan | R: reset view | Esc: quit"
        self.screen.blit(self.font.render(help_text, True, self.WHITE), (14, 12))
        if info:
            line = f"Progress: {float(info.get('progress', 0.0)):.3f} laps"
            self.screen.blit(self.font.render(line, True, self.WHITE), (14, 38))
        pygame.display.flip()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle viewer controls; return False when the viewer should close."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            self.center = np.array([event.w / 2.0, event.h / 2.0], dtype=np.float64)
            self._background = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r:
                self.zoom = 1.0
                self.pan[:] = 0.0
            elif event.key == pygame.K_LEFT:
                self.pan[0] += 35
            elif event.key == pygame.K_RIGHT:
                self.pan[0] -= 35
            elif event.key == pygame.K_UP:
                self.pan[1] += 35
            elif event.key == pygame.K_DOWN:
                self.pan[1] -= 35
        elif event.type == pygame.MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.12 ** event.y), 0.35, 4.0))
            self._background = None
        return True

    def run(self, state: Any | None = None) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self.draw(state)
            self.clock.tick(60)
        self.close()
