import itertools as it
from random import choice
from time import sleep
import vizdoom as vzd
import skimage
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CNNLSTMPolicy
import tensorflow as tf

def preprocess(img, resolution=(84, 84)):
    return np.asarray(skimage.transform.resize(img, resolution).astype(np.float32))

if __name__ == "__main__":

    game = vzd.DoomGame()
    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overwrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.
    game.load_config("../scenarios/my_way_home.cfg")

    # Makes the screen bigger to see more details.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.add_game_args('+"bind w +forward" +"bind s +back" +"bind a +left" +"bind d +right"')

    # game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    # Creates all possible actions depending on how many buttons there are.
    actions_num = game.get_available_buttons_size()
    print ("Number of actions: ", actions_num)
    print ("""
    available_buttons = 
        { 
            TURN_LEFT
            TURN_RIGHT
            MOVE_FORWARD 
            MOVE_LEFT
            MOVE_RIGHT
        }
    """)

    actions = [list(a) for a in it.product([0, 1], repeat=actions_num)]

    print ("****" * 10)
    print ("Length combination of actions: ", len(actions))
    assert 2**actions_num == len(actions)
    print ("****" * 10)

    cnnlstm_model = CNNLSTMPolicy(state_shape=[84, 84, 3], num_action=actions_num)

    episodes = 10
    sleep_time = 0.028

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(episodes):
            print("Episode #" + str(i + 1))

            # Not needed for the first episode but the loop is nicer.
            game.new_episode()
            while not game.is_episode_finished():

                # Gets the state and possibly to something with it
                state = game.get_state()
                __state = preprocess(state.screen_buffer) #numpy array with shape (84,84,3)

                logits = sess.run([cnnlstm_model.logits], feed_dict={cnnlstm_model.input: [__state] })
                print ("Logits: ", logits)

                # Makes a random action and save the reward.
                reward = game.make_action(choice(actions))

                print("Game Variables:", state.game_variables)
                print("Performed action:", game.get_last_action())
                print("Last Reward:", reward)
                print("=====================")

                # Sleep some time because processing is too fast to watch.
                if sleep_time > 0:
                    sleep(sleep_time)

            print("Episode finished!")
            print("total reward:", game.get_total_reward())
            print("************************")