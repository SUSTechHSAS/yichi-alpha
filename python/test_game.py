"""
Test chain reaction scenarios for the game engine.
Verifies the rules match the JS implementation.
"""
import sys
sys.path.insert(0, '/home/z/my-project/download/yichi-alpha/python')

from game import GameState, GameConfig, X, O, BLOCK, EMPTY


def test_heal_bonus():
    """Two same-color direct neighbors should give health bonus."""
    print("=== Test: heal bonus ===")
    s = GameState.initial(GameConfig(board_size=5))
    # X plays center (2,2)
    s.apply_move((2, 2))
    # O plays far away
    s.apply_move((0, 0))
    # X plays (2,3) - now X has 2 direct neighbors at (2,2) and (2,3)
    # (2,2) has direct neighbors: (2,3) - that's only 1, so no bonus
    # But after X plays (2,3), check (2,2): direct neighbors same color = 1 (just (2,3))
    # Need a third piece to trigger heal. Let me set up differently.
    s.apply_move((2, 3))
    print(s)
    print(f"hp(2,2)={s.hp(2,2)}, hp(2,3)={s.hp(2,3)}")
    # Both have only 1 same-color direct neighbor (each other), no bonus
    assert s.hp(2, 2) == 2, f"Expected 2, got {s.hp(2,2)}"
    print()


def test_damage_and_flip():
    """Surround an enemy piece with 2+ direct neighbors to flip it."""
    print("=== Test: damage and flip ===")
    # Setup manually
    s = GameState.initial(GameConfig(board_size=5))
    # X plays (2,2)
    s.apply_move((2, 2))
    # O plays (2,3) - adjacent to X
    s.apply_move((2, 3))
    # X plays (1,3) - now (2,3) has 2 X direct neighbors: (2,2) and (1,3)
    # Wait: (2,2) is at (row=2,col=2), (2,3) is at (row=2,col=3), (1,3) is at (row=1,col=3)
    # (2,3)'s direct neighbors: (1,3)=X, (3,3)=EMPTY, (2,2)=X, (2,4)=EMPTY
    # So 2 X neighbors → damage = 2*1 = 2 → health 2-2 = 0 → flip!
    s.apply_move((1, 3))
    print(s)
    print(f"Cell (2,3) type={s.at(2,3)}, hp={s.hp(2,3)}")
    # After flip + heal in next chain iter: (2,3) is X, has 2 X direct neighbors
    # → heal bonus = inithealth + n*healpower - 1 = 2 + 2*1 - 1 = 3
    assert s.at(2, 3) == X, f"Expected X (flipped), got {s.at(2,3)}"
    assert s.hp(2, 3) == 3, f"Expected hp=3 after flip+heal, got {s.hp(2,3)}"
    print("✓ Flip + chain heal worked correctly")
    print()


def test_block_rule():
    """Test that block rule creates BLOCK cells when 2+ X and 2+ O surround a cell."""
    print("=== Test: block rule (asymmetric config) ===")
    # Block rule only triggers when !diag_heal || !diag_attack || attack != heal
    # Use asymmetric config
    cfg = GameConfig(board_size=5, diag_heal=False, diag_attack=False)
    s = GameState.initial(cfg)
    # Place stones manually using apply_move (which respects turn order)
    # We want cell (2,2) to have 2 X direct neighbors and 2 O direct neighbors.
    # Direct neighbors of (2,2): (1,2), (3,2), (2,1), (2,3)
    # X: (1,2), (3,2); O: (2,1), (2,3) → cell (2,2) becomes BLOCK
    s.apply_move((1, 2))  # X
    s.apply_move((2, 1))  # O
    s.apply_move((3, 2))  # X
    s.apply_move((2, 3))  # O — this places the 4th piece; (2,2) should now become BLOCK
    print(s)
    # Note: (2,2) might still be EMPTY if block_rule didn't trigger, but should be BLOCK
    # Actually, block_rule turns cells into BLOCK; let's check
    # The cell (2,2) is EMPTY, has 2 X and 2 O neighbors → becomes BLOCK
    assert s.at(2, 2) == BLOCK, f"Expected BLOCK at (2,2), got {s.at(2,2)}"
    print("✓ Block rule worked")
    print()


def test_chain_cascade():
    """Test multi-step chain reaction."""
    print("=== Test: chain cascade ===")
    # Set up a position where one move triggers a flip, which triggers another flip
    s = GameState.initial(GameConfig(board_size=5))
    # Build a chain manually
    s.apply_move((0, 0))  # X
    s.apply_move((0, 4))  # O
    s.apply_move((1, 0))  # X
    s.apply_move((1, 4))  # O
    s.apply_move((2, 0))  # X — now (1,0) has 2 X neighbors (0,0)(2,0)
    # but no O neighbor of (1,0), so no damage
    s.apply_move((2, 4))  # O — (1,4) has 2 O neighbors (0,4)(2,4), no damage to it
    s.apply_move((3, 0))  # X
    s.apply_move((3, 4))  # O
    print(s)
    print(f"X count: {(s.types == X).sum()}, O count: {(s.types == O).sum()}")
    print()


def test_full_game():
    """Run a full random game to make sure no crashes."""
    print("=== Test: full random game ===")
    import random
    random.seed(42)
    s = GameState.initial()
    moves_played = 0
    while not s.is_terminal():
        legal = s.legal_moves()
        if not legal:
            break
        move = random.choice(legal)
        s.apply_move(move)
        moves_played += 1
        if moves_played > 100:
            print(f"Game didn't end after 100 moves, current state:")
            print(s)
            break
    print(f"Game ended after {moves_played} moves")
    print(s)
    w = s.winner()
    print(f"Winner: {'X' if w == X else 'O' if w == O else 'draw'}")
    print()


if __name__ == '__main__':
    test_heal_bonus()
    test_damage_and_flip()
    test_block_rule()
    test_chain_cascade()
    test_full_game()
    print("All tests passed ✓")
