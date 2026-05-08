"""Main training loop: alternate self-play and learning.

for iter in range(N):
    for _ in range(games_per_iter):
        replay.add_game(selfplay.play_game(network, cfg))
    for _ in range(train_steps_per_iter):
        train_step(network, optimizer, replay.sample(batch_size))
    if iter % checkpoint_every == 0:
        save_checkpoint(network, iter)
        log(arena.play_match(network_policy, random_policy, 20))

NOT YET IMPLEMENTED.
"""
