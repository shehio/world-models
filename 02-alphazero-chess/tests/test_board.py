"""Round-trip tests for board / move encoding.

Plan:
  - encode_board returns shape (19, 8, 8), dtype float32, values in {0, 1} except no-progress plane.
  - For every legal move on a sampled set of positions: decode_move(encode_move(m)) == m.
  - legal_move_mask matches set(encode_move(m) for m in board.legal_moves).
  - Color flipping: encode_board(white-to-move) and encode_board(same position mirrored, black-to-move)
    should look like rotations of each other in the appropriate planes.

NOT YET IMPLEMENTED.
"""
