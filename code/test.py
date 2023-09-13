import matplotlib.pyplot as plt
import pickle
with open('D:/Lab/RL/AODAI_RL/code/dqn_train0.pkl', 'rb') as file:
    dqn_train0 = pickle.load(file)
with open('code/dqn_test0.pkl', 'rb') as file:
    dqn_test0 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/actor_train0.pkl', 'rb') as file:
    actor_train0 = pickle.load(file)
with open('code/actor_test0.pkl', 'rb') as file:
    actor_test0 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/actor_train2.pkl', 'rb') as file:
    actor_train2 = pickle.load(file)
with open('code/actor_test2.pkl', 'rb') as file:
    actor_test2 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train1.pkl', 'rb') as file:
    dqn_train1 = pickle.load(file)
with open('code/dqn_test1.pkl', 'rb') as file:
    dqn_test1 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train2.pkl', 'rb') as file:
    dqn_train2 = pickle.load(file)
with open('code/dqn_test2.pkl', 'rb') as file:
    dqn_test2 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train3.pkl', 'rb') as file:
    dqn_train3 = pickle.load(file)
with open('code/dqn_test3.pkl', 'rb') as file:
    dqn_test3 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train4.pkl', 'rb') as file:
    dqn_train4 = pickle.load(file)
with open('code/dqn_test4.pkl', 'rb') as file:
    dqn_test4 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train5.pkl', 'rb') as file:
    dqn_train5 = pickle.load(file)
with open('code/dqn_test5.pkl', 'rb') as file:
    dqn_test5 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train7.pkl', 'rb') as file:
    dqn_train7 = pickle.load(file)
with open('code/dqn_test7.pkl', 'rb') as file:
    dqn_test7 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train6.pkl', 'rb') as file:
    dqn_train6 = pickle.load(file)
with open('code/dqn_test6.pkl', 'rb') as file:
    dqn_test6 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train8.pkl', 'rb') as file:
    dqn_train8 = pickle.load(file)
with open('code/dqn_test8.pkl', 'rb') as file:
    dqn_test8 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train9.pkl', 'rb') as file:
    dqn_train9 = pickle.load(file)
with open('code/dqn_test9.pkl', 'rb') as file:
    dqn_test9 = pickle.load(file)
with open('D:/Lab/RL/AODAI_RL/code/dqn_train10.pkl', 'rb') as file:
    dqn_train10 = pickle.load(file)
with open('code/dqn_test10.pkl', 'rb') as file:
    dqn_test10 = pickle.load(file)
actor_train2=[x*-800 for x in actor_train2]
actor_test2=[x*-800 for x in actor_test2]
# #30 turn upgrade
# plt.plot(dqn_train0[1000:],'r')
# #normal
# plt.plot(dqn_train1,'y')
# #tau=0.5
# plt.plot(dqn_train2[1000:],'g')
# #200 turn
# plt.plot(dqn_train3[1000:],'c')
# #nomal 2
# plt.plot(dqn_train4,'b')
# #decay tau
# plt.plot(dqn_train6,'k')
# #decay tau +30 turn
# plt.plot(dqn_train7,'r')
##100turn
# plt.plot(dqn_train5[1000:])
# #decay start=0.5+30 turn
# plt.plot(dqn_train8,'c')
# #decay start=0.4+15 turn
# plt.plot(dqn_train8,'c')

# plt.plot(dqn_train0,'r')
# plt.plot(actor_train1,'b')
# plt.plot(dqn_train1,'y')
# plt.plot(dqn_train2,'g')

plt.plot(dqn_test4,'r')
# plt.plot(dqn_test10,'b')
# plt.plot(dqn_test7,'y')
plt.plot(dqn_test6,'g')
# plt.plot(dqn_test8,'c')

plt.xlabel('time_slot')
plt.ylabel('Time per all task')
plt.show()