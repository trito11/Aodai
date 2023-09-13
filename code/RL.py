from main import Run_ExpectedTaskDDQN

if __name__ == "__main__":
    for i in range(1, 2):
        try:
            Run_ExpectedTaskDDQN("test/" + str(i), gamma=0.9,
                                 target_model_update=1e-2)
        except Exception as e:
            print(e)
