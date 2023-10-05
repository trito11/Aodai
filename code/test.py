import matplotlib.pyplot as plt
import pickle

with open('D:/Lab/RL/AODAI_RL/code/dqn_train_notrans1.pkl', 'rb') as file:
    train_no_trans1 = pickle.load(file)
with open('code/dqn_test_notrans1.pkl', 'rb') as file:
    test_no_trans1 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans2.pkl', 'rb') as file:
    train_trans2_ema = pickle.load(file)
with open('code/dqn_test_trans2.pkl', 'rb') as file:
    test_trans2_ema = pickle.load(file)
with open('code/dqn_test_notrans1.pkl', 'rb') as file:
    test_no_trans1 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans3.pkl', 'rb') as file:
    train_trans3_ema = pickle.load(file)
with open('code/dqn_test_trans3.pkl', 'rb') as file:
    test_trans3_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans4.pkl', 'rb') as file:
    train_trans4_ema = pickle.load(file)
with open('code/dqn_test_trans4.pkl', 'rb') as file:
    test_trans4_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans6.pkl', 'rb') as file:
    train_trans6_ema = pickle.load(file)
with open('code/dqn_test_trans6.pkl', 'rb') as file:
    test_trans6_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans7.pkl', 'rb') as file:
    train_trans7_ema = pickle.load(file)
with open('code/dqn_test_trans7.pkl', 'rb') as file:
    test_trans7_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans8.pkl', 'rb') as file:
    train_trans8_ema = pickle.load(file)
with open('code/dqn_test_trans8.pkl', 'rb') as file:
    test_trans8_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_trans_no_ema.pkl', 'rb') as file:
    train_trans_no_ema = pickle.load(file)
with open('code/dqn_test_trans_no_ema.pkl', 'rb') as file:
    test_trans_no_ema = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_notrans2.pkl', 'rb') as file:
    train_no_trans2 = pickle.load(file)
with open('code/dqn_test_notrans2.pkl', 'rb') as file:
    test_no_trans2 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train_notrans3.pkl', 'rb') as file:
    train_no_trans3 = pickle.load(file)
with open('code/dqn_test_notrans3.pkl', 'rb') as file:
    test_no_trans3 = pickle.load(file)
def create_new_array(arr, window_size):
    new_arr = []
    for i in range(0, len(arr), window_size):
        window = arr[i:i + window_size]
        window_sum = sum(window)
        new_arr.append(window_sum)
    return new_arr
window_size = 121
no_trans1=create_new_array( train_no_trans1,window_size)
no_trans2=create_new_array( train_no_trans2,window_size)
no_trans3=create_new_array( train_no_trans3,window_size)
trans=create_new_array( train_trans2_ema,window_size)
trans1=create_new_array( train_trans3_ema,window_size)
trans2=create_new_array( train_trans4_ema,window_size)
trans3=create_new_array( train_trans6_ema,window_size)
trans4=create_new_array( train_trans7_ema,window_size)
trans5=create_new_array( train_trans8_ema,window_size)
tran_no_ema=create_new_array( train_trans_no_ema,window_size)

# plt.plot(test_no_trans2,'g',label='dqn2')
# plt.plot(test_no_trans1,'r',label='dqn1')
# plt.plot(test_trans3_ema,'y',label='dqn+trans+ema1')
# plt.plot(test_trans4_ema,'b',label='dqn+trans+ema2')


print(f'no_trans1: {sum(no_trans1)/1089},no_trans2: {sum(no_trans2)/1089}, trans: {sum(trans)/1089}, trans1: {sum(trans1)/1089}, tran_no_ema: {sum(tran_no_ema)/1089}')



plt.plot(no_trans3,'r',label="dqn1")
plt.plot(trans3,'b',label='dqn+trans+ema')
plt.plot(trans4,'y',label='dqn+trans+ema+layer2')
plt.plot(trans5,'c',label='dqn+trans+ema+layer2,3')
# plt.plot(trans2,'g',label='dqn+trans+ema3')


plt.xlabel('time_slot')
plt.ylabel('Sum reward every 33 episodes')
plt.legend()
plt.show()

# tran6 ema
# tran7 freezelayer2
# trans8 freeze layer2+3