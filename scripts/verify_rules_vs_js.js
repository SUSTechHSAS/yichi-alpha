// verify_rules_vs_js.js  (v2)
// Extract the rule functions from the original JS source, run them headless,
// generate random games, and dump (board, move) sequences for comparison.

const fs = require('fs');

// Read the original game source
const src = fs.readFileSync('/home/z/my-project/scripts/game_source.js', 'utf8');

// Extract just the function definitions we need (they don't depend on DOM)
const needed = [
    'BOARD_SIZE', 'EMPTY', 'PLAYER_X', 'PLAYER_O', 'BLOCK',
    'inithealth', 'attackpower', 'healpower', 'blockdensity',
    'isDiagHeal', 'isDiagAttack',
];

// Strategy: build a self-contained module that includes:
//   1. State variable declarations (with init)
//   2. All rule functions (initializeBoard, refreshBoard, healRule, damageRule,
//      deathRule, blockRule, isBoardFull, countPieces, determineWinner)
//   3. A custom applyMove function that uses the same logic as handleClick
//      but without DOM dependencies.

// Find each function by name in the source and extract its body
function extractFunction(name) {
    // Match: function name(...) { ... }  with balanced braces
    const start = src.indexOf(`function ${name}(`);
    if (start === -1) throw new Error(`Function ${name} not found`);
    let i = src.indexOf('{', start);
    let depth = 1;
    i++;
    while (depth > 0 && i < src.length) {
        const c = src[i];
        if (c === '{') depth++;
        else if (c === '}') depth--;
        i++;
    }
    return src.slice(start, i);
}

const funcsToExtract = [
    'initializeBoard',
    'isBoardFull',
    'countPieces',
    'determineWinner',
    'healRule',
    'damageRule',
    'deathRule',
    'blockRule',
    'refreshBoard',
    'deepCopyBoard',
    'checkEasterEggPattern',
];

let extracted = '';
for (const f of funcsToExtract) {
    let body = extractFunction(f);
    // Strip async/await (refreshBoard uses await for delay rendering, not needed for logic)
    body = body.replace(/async function/g, 'function');
    body = body.replace(/await new Promise[^;]+;/g, '/* await stripped */');
    extracted += body + '\n\n';
}

// Build the standalone module
const moduleCode = `
// Auto-extracted from original 异吃棋 source
let BOARD_SIZE = 6;
const EMPTY = '-';
const PLAYER_X = 'x';
const PLAYER_O = 'o';
const BLOCK = '☒';
let inithealth = 2;
let attackpower = 1;
let healpower = 1;
let blockdensity = 0;
let isDiagHeal = true;
let isDiagAttack = true;
let isPrioUs = false;
let isDelayEnabled = false;
let board = [];
let currentPlayer = PLAYER_X;
let boardChanged = false;
let hasQed = true;  // start true to skip easter egg check

// Stub document so checkEasterEggPattern doesn't crash (we set hasQed=true above to skip it)
global.document = { getElementById: () => ({ innerHTML: '' }) };

${extracted}

// Custom applyMove that mimics handleClick but without DOM
function applyMove(row, col) {
    if (board[row][col].type !== EMPTY) return false;
    board[row][col].type = currentPlayer;
    board[row][col].health = 2;
    do {
        boardChanged = false;
        refreshBoard();
    } while (boardChanged);
    if (isBoardFull()) return true;
    currentPlayer = (currentPlayer === PLAYER_X) ? PLAYER_O : PLAYER_X;
    return true;
}

function getBoardState() {
    return {
        types: board.map(row => row.map(cell => cell.type)),
        health: board.map(row => row.map(cell => cell.health)),
        currentPlayer: currentPlayer,
    };
}

function resetBoard(size) {
    BOARD_SIZE = size;
    board = [];
    for (let i = 0; i < BOARD_SIZE; i++) {
        board[i] = [];
        for (let j = 0; j < BOARD_SIZE; j++) {
            board[i][j] = { type: EMPTY, health: 0 };
        }
    }
    currentPlayer = PLAYER_X;
    boardChanged = false;
    hasQed = false;
}

function legalMoves() {
    // Mirror python implementation: empty cells within 2-Chebyshev of any piece
    const moves = [];
    let hasPiece = false;
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j].type !== EMPTY) { hasPiece = true; break; }
        }
        if (hasPiece) break;
    }
    if (!hasPiece) {
        for (let i = 0; i < BOARD_SIZE; i++)
            for (let j = 0; j < BOARD_SIZE; j++)
                moves.push([i, j]);
        return moves;
    }
    const seen = new Set();
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j].type === EMPTY) continue;
            for (let di = -2; di <= 2; di++) {
                for (let dj = -2; dj <= 2; dj++) {
                    const ni = i + di, nj = j + dj;
                    if (ni >= 0 && ni < BOARD_SIZE && nj >= 0 && nj < BOARD_SIZE
                        && board[ni][nj].type === EMPTY) {
                        const key = ni * BOARD_SIZE + nj;
                        if (!seen.has(key)) {
                            seen.add(key);
                            moves.push([ni, nj]);
                        }
                    }
                }
            }
        }
    }
    return moves;
}

module.exports = {
    resetBoard, applyMove, getBoardState, legalMoves, isBoardFull, countPieces,
};
`;

const tmpFile = '/tmp/yichi_orig_module.js';
fs.writeFileSync(tmpFile, moduleCode);

try {
    const game = require(tmpFile);
    console.log('=== Original JS module loaded successfully ===');
    console.log('Running 50 random games for verification...\n');

    // Generate 50 random games with deterministic seeds
    const games = [];
    for (let g = 0; g < 50; g++) {
        game.resetBoard(6);
        const trajectory = [];
        let seed = g * 7919 + 12345;
        function rng() {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return seed;
        }
        let steps = 0;
        while (!game.isBoardFull() && steps < 50) {
            const state = game.getBoardState();
            const moves = game.legalMoves();
            if (moves.length === 0) break;
            const move = moves[rng() % moves.length];
            trajectory.push({ state, move });
            game.applyMove(move[0], move[1]);
            steps++;
        }
        const finalState = game.getBoardState();
        const counts = game.countPieces();
        games.push({ trajectory, finalState, counts });
        if (g % 10 === 9) console.log(`  Game ${g+1}/50 done`);
    }

    // Dump to JSON for comparison
    fs.writeFileSync('/tmp/yichi_js_games.json', JSON.stringify(games));
    console.log(`\nDumped ${games.length} games to /tmp/yichi_js_games.json`);
} catch (e) {
    console.error('Failed:', e.message);
    console.error(e.stack);
    process.exit(1);
}
