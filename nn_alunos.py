"""Colocar o nome dos elementos do grupo"""

import random
import math

#valor exemplificativo
new = 0.6
tipo_animal =  [ "mammal", "bird", "reptile", "fish","amphibian","insect","invertebrate" ]


def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    
    
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(input):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(- input))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada in (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    
    #copia a informacao do vector de entrada in para a listavector de inputs da rede nn  
    nn['x']=input.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [[w+x*nn['dz'][i]*new for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*new for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    
    forward(nn, input)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    


def train_and():
    """Funcao que cria uma rede 2x2x1 e treina um AND"""
    
    net = make(2, 2, 1)
    for i in range(2000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
def train_or():
    """Funcao que cria uma rede 2x2x1 e treina um OR"""
    
    net = make(2, 2, 1)
    for i in range(1000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

def train_xor():
    """Funcao que cria uma rede 2x2x1 e treina um XOR"""
    
    net = make(2, 2, 1)
    for i in range(10000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net
    


def run():
    """Funcao principal do nosso programa, cria os conjuntos de treino e teste, chama
    a funcao que cria e treina a rede e, por fim, a funcao que a treina"""
    
    pass
    


def build_sets(f):
    """Funcao que cria os conjuntos de treino e de de teste a partir dos dados
    armazenados em f (zoo.txt). A funcao le cada linha, tranforma-a numa lista
    de valores e chama a funcao translate para a colocar no formato adequado para
    o padrao de treino. Estes padroes são colocados numa lista 
    Finalmente, devolve duas listas, uma com os primeiros 67 padroes (conjunto de treino)
    e a segunda com os restantes (conjunto de teste)"""
    lista = list()
    with open(f, 'r',encoding="utf-8") as file:
        [lista.append(line.strip().strip('[').strip(']').split(',')) for line in file.readlines()]
        
    translate(lista)
    
    
    return lista[:67],lista[67:]


def translate(lista):
    """Recebe cada lista de valores e transforma-a num padrao de treino.
    Cada padrao tem o formato [nome_do_animal, padrao_de_entrada, tipo_do_animal, padrao_de_saida].
    nome_do_animal e o primeiro valor da lista e tipo_de_animal o ultimo.
    padrao_de_entrada e uma lista de 0 e 1 com os valores dos atributos.
    O numero de pernas deve tambem ser convertido numa lista de 0 e 1, concatenada com os restantes
    atributos. E.g. [0 0 0 0 1 0 0 0 0 0] -> 4 pernas.
    padrao_de_saida e uma lista de 0 e 1 que representa o tipo do animal. Tem 7 posicoes e a unica
    que estiver a 1 corresponde ao tipo do animal. E.g., [0 0 1 0 0 0 0] -> reptile.
    """
    for i in range(len(lista)):
        nova_lista = list()
               
        
        for j in range(len(lista[i])):
            if j == 13:
                padrao = [0] * 10
                padrao[int(lista[i][13])] = 1
        
                nova_lista += padrao                
            elif lista[i][j].isdigit():
                nova_lista.append(int(lista[i][j]))
        del lista[i][1:17]
        lista[i].insert(1,nova_lista)
        
        tipo = [0] * 7
        tipo[tipo_animal.index(lista[i][2])] = 1

        lista[i].insert(3,tipo)

    random.shuffle(lista)
    pass
        

def train_zoo(training_set):
    """cria a rede e chama a funçao iterate para a treinar. Use 300 iteracoes"""
    #training_set[1][1] == input training_set[1][3] == output 
    #iterate(i, nn, input, output)
    net = make(25, 25, 1)
    for i in range(300):
        [iterate(i, net, training_set[j][1], training_set[j][3]) for j in range(len(training_set))]
    return net

def retranslate(out):
    """recebe o padrao de saida da rede e devolve o tipo de animal corresponte.
    Devolve o tipo de animal corresponde ao indice da saida com maior valor."""
    return tipo_animal[out.index(1)]

def test_zoo(net, test_set):
    """Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste.
    Para cada padrao do conjunto de teste chama a funcao forward e determina o tipo
    do animal que corresponde ao maior valor da lista de saida. O tipo determinado
    pela rede deve ser comparado com o tipo real, sendo contabilizado o número
    de respostas corretas. A função calcula a percentagem de respostas corretas"""
    maxY = 0
    nn_saida = list()
    for i in range(len(test_set)):
        forward(net, test_set[i][1])
        nn_saida.append(net['y'])
        #retranslate(test_set[i][3])
        #if net['y'][0] > maxY:
            #maxY = net['y'][0]
            #lista_saida.clear()
            #lista_saida += test_set[i][3]
    print(test_set[nn_saida.index(max(nn_saida))])
    
    pass

if __name__ == "__main__":
    #train_and()
    conjunto_treino,conjunto_testes = build_sets('zoo.txt')
    n = train_zoo(conjunto_testes)
    test_zoo(n,conjunto_testes) 
    #run()