import numpy as np
import gym
import random

env = gym.make("Taxi-v2") #Define o ambiente
action_size = env.action_space.n #Define o número de ações
state_size = env.observation_space.n #Define o número de estados
qtable = np.zeros((state_size, action_size)) #Criação da Qtable de acordo com o número de ações e estados

total_episodes = 50000 #Número de iterações para treinar o algoritmo
total_test_episodes = 100 #Número de iterações para testes
max_steps = 99 #Número de passos máximo por iteração

learning_rate = 0.7 #Taxa de aprendizado (Define o peso da recompensa de cada ação, portanto quanto mais próximo de 1 maior será o valor da recompensa por uma ação)
gamma = 0.618 #Fator de desconto (Quantifica a importância de recompensas futuras. Quanto mais próximo de 1, o algoritmo estará mais disposto a atrasar uma recompensa para que ela seja maior no futuro)

#Parâmetros de exploração
epsilon = 1.0 #Taxa de exploração desejada (Se a recompensa estimada for menor que este valor, a exploração terá preferência ao invés da reclamação da recompensa)
max_epsilon = 1.0 #Taxa de exploração máxima
min_epsilon = 0.01 #Taxa de exploração mínima
decay_rate = 0.01 #Taxa de subtração da exploração

#Repetição até que o número máximo de passos seja atingido
for episode in range(total_episodes):
    
    #Reinicia o ambiente
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1) #Estipula a recompensa imediata
        
        #Se a recompensa for maior que a exploração, aproveite-a
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        #Se não, explore
        else:
            action = env.action_space.sample()

        #Realiza a ação e pega os resultados obtidos dela
        new_state, reward, done, info = env.step(action)

        #Atualiza a Qtable de acordo com os dados obtidos
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        state = new_state #Atualiza o estado
        
        #Caso o aprendizado já tenha acabado, isto é, a Qtable está completamente preenchida, este passo acabada
        if done == True: 
            break

    episode += 1 

    #Reduz a exploração
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("-------------------------------------------------------")
    print("ITERACAO ", episode+1)

    for step in range(max_steps):
        env.render()
        
        #Pega a ação que dará a maior recompensa para o estado atual
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            print("\nPontos: ", total_rewards)
            break
        state = new_state
env.close()