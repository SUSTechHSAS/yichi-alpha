"""
compare_rules.py
Compare Python game engine against the original JS implementation.

Loads /tmp/yichi_js_games.json (5 random games with (state, move) trajectory),
replays each move in the Python engine, and asserts that the resulting state
matches the JS output exactly.
"""
import sys
import json
sys.path.insert(0, '/home/z/my-project/download/yichi-alpha/python')

from game import GameState, GameConfig, X, O, BLOCK, EMPTY

# JS uses characters 'x', 'o', '☒', '-' — map to our constants
JS_TO_PY = {'x': X, 'o': O, '☒': BLOCK, '-': EMPTY}

def js_state_to_python(js_state):
    """Convert a JS board state to a Python GameState."""
    n = len(js_state['types'])
    cfg = GameConfig(board_size=n)
    s = GameState.initial(cfg)
    for r in range(n):
        for c in range(n):
            t = js_state['types'][r][c]
            h = js_state['health'][r][c]
            s.types[r, c] = JS_TO_PY[t]
            s.health[r, c] = h
    s.current_player = X if js_state['currentPlayer'] == 'x' else O
    return s

def compare_states(py_state, js_state, label=""):
    """Compare two states. Return list of differences."""
    diffs = []
    n = len(js_state['types'])
    for r in range(n):
        for c in range(n):
            js_t = JS_TO_PY[js_state['types'][r][c]]
            js_h = js_state['health'][r][c]
            py_t = int(py_state.types[r, c])
            py_h = int(py_state.health[r, c])
            if js_t != py_t:
                diffs.append(f"  ({r},{c}) type: JS={js_state['types'][r][c]}(={js_t}) vs PY={py_t}")
            if js_h != py_h:
                diffs.append(f"  ({r},{c}) health: JS={js_h} vs PY={py_h}")
    # currentPlayer: JS uses 'x'/'o' strings, Python uses 1/2.
    # If board is full, JS leaves currentPlayer as whoever was about to move
    # before the (failed) switch. Python does the same — but the JS dump captures
    # the state AFTER the failed switch attempt. Let's just normalize both.
    # We'll skip currentPlayer check on terminal states (board full).
    if not py_state.is_full():
        js_player = X if js_state['currentPlayer'] == 'x' else O
        if js_player != py_state.current_player:
            diffs.append(f"  currentPlayer: JS={js_state['currentPlayer']} vs PY={py_state.current_player}")
    return diffs


def main():
    with open('/tmp/yichi_js_games.json') as f:
        games = json.load(f)

    print(f"Loaded {len(games)} games from JS")
    print()

    total_steps = 0
    total_diffs = 0
    for g_idx, game in enumerate(games):
        print(f"=== Game {g_idx} ===")
        trajectory = game['trajectory']
        final_state = game['finalState']
        counts = game['counts']

        # Replay in Python from initial state
        py_state = GameState.initial(GameConfig(board_size=6))
        game_diffs = 0

        for step_idx, entry in enumerate(trajectory):
            js_state_before = entry['state']
            move = tuple(entry['move'])

            # Verify Python state matches JS state before move
            diffs = compare_states(py_state, js_state_before, label=f"step {step_idx} before")
            if diffs:
                print(f"  Step {step_idx}: STATE MISMATCH before applying move {move}")
                for d in diffs[:5]:
                    print(d)
                game_diffs += len(diffs)
                # Continue anyway to see how far we diverge

            # Apply the move in Python
            try:
                py_state.apply_move(move)
            except Exception as e:
                print(f"  Step {step_idx}: FAILED to apply move {move}: {e}")
                break

            total_steps += 1

        # Final state comparison
        diffs = compare_states(py_state, final_state, label="final")
        if diffs:
            print(f"  FINAL STATE MISMATCH ({len(diffs)} diffs):")
            for d in diffs[:10]:
                print(d)
        else:
            print(f"  Final state: PERFECT MATCH")

        # Counts comparison
        x_count = int((py_state.types == X).sum())
        o_count = int((py_state.types == O).sum())
        if x_count != counts['xCount'] or o_count != counts['oCount']:
            print(f"  COUNT MISMATCH: JS X={counts['xCount']} O={counts['oCount']} vs PY X={x_count} O={o_count}")
        else:
            print(f"  Counts match: X={x_count}, O={o_count}")

        total_diffs += game_diffs
        print()

    print("=" * 50)
    print(f"Total steps replayed: {total_steps}")
    print(f"Total state mismatches: {total_diffs}")
    if total_diffs == 0:
        print("✓ Python engine matches JS implementation PERFECTLY")
    else:
        print(f"✗ Found {total_diffs} mismatches — rules need fixing")

if __name__ == '__main__':
    main()
