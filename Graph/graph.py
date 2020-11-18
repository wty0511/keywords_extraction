import networkx as nx

class graph:

    def __init__(self,
                 word_list,
                 window=10,

                 ):
        self.graph = nx.Graph()
        self.word_list = word_list
        self.window = window
        self.build_graph()
    def build_graph(self):

        words = self.word_list

        for j in range(len(words)):
            #print(words[j])


            if not self.graph.has_node(words[j]):
                self.graph.add_node(words[j])
                #print('add')


        for i in range(len(words)):

            word1 = words[i]

            for k in range(i + 1, min(len(words), i + self.window)):
                word2 = words[k]
                if not self.graph.has_edge(word1, word2):
                    self.graph.add_edge(word1, word2, weight=0)
                self.graph[word1][word2]['weight'] += 1

    def centrality(self):
        print(self.graph.edges())
        return nx.degree_centrality(self.graph),nx.closeness_centrality(self.graph),nx.betweenness_centrality(self.graph),nx.eigenvector_centrality_numpy(self.graph)








