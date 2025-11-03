import copy
import sys

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def is_terminal(board):
    return check_winner(board, "X") or check_winner(board, "O") or all(cell != " " for row in board for cell in row)

def check_winner(board, player):
    win_states = [
        [(0,0), (0,1), (0,2)],
        [(1,0), (1,1), (1,2)],
        [(2,0), (2,1), (2,2)],
        [(0,0), (1,0), (2,0)],
        [(0,1), (1,1), (2,1)],
        [(0,2), (1,2), (2,2)],
        [(0,0), (1,1), (2,2)],
        [(0,2), (1,1), (2,0)]
    ]
    return any(all(board[r][c] == player for r, c in line) for line in win_states)

def utility(board):
    if check_winner(board, "X"):
        return 1
    elif check_winner(board, "O"):
        return -1
    else:
        return 0

def get_legal_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]

def minimax(board, is_maximizing):
    if is_terminal(board):
        return utility(board), None

    best_move = None
    if is_maximizing:
        max_eval = float("-inf")
        for move in get_legal_moves(board):
            new_board = copy.deepcopy(board)
            new_board[move[0]][move[1]] = "X"
            eval, _ = minimax(new_board, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in get_legal_moves(board):
            new_board = copy.deepcopy(board)
            new_board[move[0]][move[1]] = "O"
            eval, _ = minimax(new_board, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

def play_game():
    board = [[" "]*3 for _ in range(3)]
    print("Welcome to Tic-Tac-Toe! You are O. Computer is X.")
    print_board(board)

    while not is_terminal(board):
        move = None
        while move not in get_legal_moves(board):
            try:
                user_input = input("Enter your move as row,col (e.g., 1,2): ")
                r, c = map(int, user_input.split(","))
                move = (r, c)
            except:
                print("Invalid input. Try again.")
        board[move[0]][move[1]] = "O"
        print("\nYour move:")
        print_board(board)

        if is_terminal(board):
            break

        _, comp_move = minimax(board, True)
        board[comp_move[0]][comp_move[1]] = "X"
        print("\nComputer's move:")
        print_board(board)

    if check_winner(board, "X"):
        print("Computer wins!")
    elif check_winner(board, "O"):
        print("You win!")
    else:
        print("It's a draw!")

def report_mode():
    """Run a quick automatic demo game for reporting."""
    board = [[" "]*3 for _ in range(3)]
    turn = "X"  # Computer starts
    print("Tic-Tac-Toe Automatic Report Mode")
    print_board(board)

    while not is_terminal(board):
        if turn == "X":
            _, move = minimax(board, True)
        else:
            _, move = minimax(board, False)

        if move:
            board[move[0]][move[1]] = turn
        turn = "O" if turn == "X" else "X"
        print()
        print_board(board)

    if check_winner(board, "X"):
        print("Computer (X) wins!")
    elif check_winner(board, "O"):
        print("Opponent (O) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    if "--report" in sys.argv:
        report_mode()
    else:
        play_game()