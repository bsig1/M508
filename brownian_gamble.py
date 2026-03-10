# brownian_gamble.py
# Proof-of-concept: bankroll follows a Brownian-style random walk.
# Press SPACE to cash out immediately. Press R to restart. ESC or close window to quit.

import random
import pygame
import pygame._freetype as freetype

# --- Settings ---
WIDTH, HEIGHT = 900, 520
FPS = 60
DT = 1 / FPS

START_BANKROLL = 100.0
MU = -3         # drift (expected direction per second)
SIGMA = 20.0     # volatility (larger = wilder swings)
MIN_BANKROLL = 0.0
GRAPH_RECT = pygame.Rect(380, 40, 500, 430)

# --- Pygame setup ---
pygame.init()
freetype.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brownian Motion Gambling POC")
clock = pygame.time.Clock()


def load_font(size):
    # pygame.font is broken on some pygame/python combos; use freetype directly.
    return freetype.Font(None, size)


font = load_font(24)
small_font = load_font(18)


def simulate_step(value, mu, sigma, dt):
    # dX = mu*dt + sigma*sqrt(dt)*N(0,1)
    shock = random.gauss(0, 1)
    return max(MIN_BANKROLL, value + mu * dt + sigma * (dt ** 0.5) * shock)


def draw_text(text, y, color=(240, 240, 240), fnt=font):
    surf, _ = fnt.render(text, color)
    screen.blit(surf, (20, y))


def draw_graph(history, rect):
    pygame.draw.rect(screen, (28, 35, 50), rect, border_radius=8)
    pygame.draw.rect(screen, (70, 85, 110), rect, 2, border_radius=8)
    if len(history) < 2:
        return

    low = min(history)
    high = max(history)
    if low == high:
        low -= 1.0
        high += 1.0

    x0, y0, w, h = rect
    points = []
    for i, value in enumerate(history):
        t = i / (len(history) - 1)
        x = x0 + int(t * (w - 1))
        y_ratio = (value - low) / (high - low)
        y = y0 + h - 1 - int(y_ratio * (h - 1))
        points.append((x, y))

    start_ratio = (START_BANKROLL - low) / (high - low)
    start_y = y0 + h - 1 - int(start_ratio * (h - 1))
    pygame.draw.line(screen, (95, 110, 135), (x0, start_y),
                     (x0 + w - 1, start_y), 1)
    pygame.draw.lines(screen, (120, 220, 255), False, points, 2)


def main():
    bankroll = START_BANKROLL
    peak = bankroll
    high_score = bankroll
    history = [bankroll]
    running = True
    cashed_out = False
    final_cashout = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not cashed_out:
                    cashed_out = True
                    final_cashout = bankroll
                    high_score = max(high_score, final_cashout)
                elif event.key == pygame.K_r:
                    bankroll = START_BANKROLL
                    peak = bankroll
                    history = [bankroll]
                    cashed_out = False
                    final_cashout = None

        if not cashed_out and bankroll > MIN_BANKROLL:
            bankroll = simulate_step(bankroll, MU, SIGMA, DT)
            peak = max(peak, bankroll)
            history.append(bankroll)

        # --- Render ---
        screen.fill((17, 23, 34))

        color = (100, 230, 130) if bankroll >= START_BANKROLL else (
            240, 120, 120)
        draw_text(f"Current bankroll: ${bankroll:,.2f}", 30, color)
        draw_text(f"Start bankroll:   ${START_BANKROLL:,.2f}", 70)
        draw_text(f"Peak bankroll:    ${peak:,.2f}", 110)
        draw_text(f"High score:       ${high_score:,.2f}", 150)
        draw_graph(history, GRAPH_RECT)

        if cashed_out:
            draw_text(f"CASHED OUT at ${
                      final_cashout:,.2f}", 190, (255, 220, 90))
            draw_text("Press R to restart, ESC to quit.",
                      230, (200, 210, 230), small_font)
        elif bankroll <= MIN_BANKROLL:
            draw_text("BUST: bankroll hit $0.00", 190, (255, 90, 90))
            draw_text("Press R to restart, ESC to quit.",
                      230, (200, 210, 230), small_font)
        else:
            draw_text("Press SPACE to cash out now.", 190, (130, 200, 255))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
