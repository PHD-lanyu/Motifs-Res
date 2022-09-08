# -- coding: utf-8 --
from baseclass.DeepRecommender import DeepRecommender
from baseclass.SocialRecommender import SocialRecommender
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import os
from tool import config
from math import sqrt
from . import layer_utils
import time

from tool.file import FileIO

from time import strftime, localtime, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class MotifsRes(SocialRecommender, DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,
                                   fold=fold)

    def readConfiguration(self):
        super(MotifsRes, self).readConfiguration()
        args = config.LineConfig(self.config['MotifsRes'])
        if self.config.contains('layer'):
            # self.layer = float(self.config['layer'])
            # self.n_layers = int(args['-n_layer'])
            self.n_layers = int(self.config['layer'])
        self.ss_rate = float(args['-ss_rate'])
        # self.node_dim = 50
        self.node_dim = int(args['-node_dim'])
        # self.perspect = 49
        self.perspect = self.node_dim - 1
        self.watch = {}
        self.match_ratio = float(args['-match_ratio'])
        self.max_drop_rate = float(args['-drop_rate'])
        self.lamda = float(args['-lamda'])
        if self.config.contains('lamda'):
            self.lamda = float(self.config['lamda'])
        if self.config.contains('ss_rate'):
            self.ss_rate = float(self.config['ss_rate'])
        if self.config.contains('ss_model_rate'):
            self.ss_model_rate = float(self.config['ss_model_rate'])
        if self.config.contains('with_mp_cosine'):
            self.with_mp_cosine = True
        else:
            self.with_mp_cosine = False
        if self.config.contains('with_cosine'):
            self.with_cosine = True
        else:
            self.with_cosine = False

        print('with_mp_cosine:{}'.format(self.with_mp_cosine))
        # print('with_mp_cosine:{}'.format(self.config.contains('with_mp_cosine')))
        print('with_mp_cosine:{}'.format(int(self.config['with_mp_cosine'])))

        if self.config.contains('with_ssl'):
            self.with_ssl = True
        else:
            self.with_ssl = False
        print('with_ssl:{}'.format(self.with_ssl))
        if self.config.contains('with_match'):
            self.with_match = True
        else:
            self.with_match = False
        # self.with_match = False
        print('with_match:{}'.format(self.with_match))
        if self.config.contains('with_split_match_ssl'):
            self.with_split_match_ssl = True
        else:
            self.with_split_match_ssl = False
        print('with_split_match:{}'.format(self.with_split_match_ssl))
        # self.drop_rate = float(args['-drop_rate'])
        # self.match_ratio=1.0

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        indices = [[self.data.user[item[0]], self.data.item[item[1]]] for item in self.data.trainingData]
        values = [float(item[2]) / sqrt(len(self.data.trainSet_u[item[0]])) / sqrt(len(self.data.trainSet_i[item[1]]))
                  for item in self.data.trainingData]
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.num_users, self.num_items])
        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10 = Y.dot(Y.T) - A8 - A9
        A10 = csr_matrix(A10)

        # A11 = (Y.T).dot(Y)
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])

        H_s = H_s.T.multiply(1.0 / (H_s.sum(axis=1) + 1e-5).reshape(1, -1))
        H_s = H_s.T

        H_j = sum([A8, A9])

        H_j = H_j.T.multiply(1.0 / (H_j.sum(axis=1) + 1e-5).reshape(1, -1))
        H_j = H_j.T

        H_p = A10.tocoo()
        r, c = H_p.row, H_p.col
        data = H_p.data
        ind = []

        # H_i = A11.tocoo()
        # r1, c1 = H_i.row, H_i.col
        # data_1 = H_i.data
        # ind_1 = []
        for i in range(data.shape[0]):  # remove pairs appearing only 1 time.
            if data[i] > 1:
                ind.append(i)
        data = [data[i] for i in ind]
        row = [r[i] for i in ind]
        col = [c[i] for i in ind]
        H_p = coo_matrix((data, (row, col)), shape=(self.num_users, self.num_users))
        # H_i = coo_matrix((H_i.data, (H_i.row, col)), shape=(self.num_users, self.num_users))
        H_p = H_p.T.multiply(1.0 / H_p.sum(axis=1).reshape(1, -1))
        H_p = H_p.T
        # return [H_s, H_j, H_p, H_i]
        return [H_s, H_j, H_p]

    def adj_to_sparse_tensor(self, adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    def initModel(self):
        super(MotifsRes, self).initModel()
        M_matrices = self.buildMotifInducedAdjacencyMatrix()
        self.weights = {}
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.glorot_uniform_initializer()
        self.channels = ['Social', 'Joint', 'Purchase']
        # self.channels = ['Social', 'Joint']
        # self.channels = ['Social', 'Purchase']
        # self.channels = ['Joint', 'Purchase']
        # self.channels = ['Social']
        # self.channels = ['Joint']
        # self.channels = ['Purchase']
        self.n_channel = 4
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.tf_is_training = tf.placeholder(tf.bool, name='tf_is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        # define learnable paramters
        for i in range(self.n_channel):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.embed_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                              name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.embed_size]),
                                                                   name='sg_W_b_%d_1' % (i + 1))
            if self.with_split_match_ssl:
                self.weights['sgating_match%d' % (i + 1)] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                                        name='sg_W_M_%d_1' % (i + 1))
                self.weights['sgating_match_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.embed_size]),
                                                                             name='sg_W_M_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.embed_size]), name='at')
        # self.weights['attention_i'] = tf.Variable(initializer([1, self.embed_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.embed_size, self.embed_size]), name='atm')
        # self.weights['attention_mat_i'] = tf.Variable(initializer([self.embed_size, self.embed_size]), name='atm')
        if self.with_match:
            self.weights['attention_match_channel'] = tf.Variable(initializer([1, self.embed_size]), name='amc')
            self.weights['attention_match_channel_mat'] = tf.Variable(initializer([self.embed_size, self.embed_size]),
                                                                      name='amcm')
        if self.with_mp_cosine:
            self.weights['match_weight'] = tf.Variable(initializer([self.perspect, self.embed_size]),
                                                       name='match_weight')

        # define inline functions
        def self_gating(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(
                tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))

        def self_supervised_gating(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(
                tf.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))

        def self_supervised_gating_for_match(em, channel):
            return tf.multiply(em, tf.nn.sigmoid(
                tf.matmul(em, self.weights['sgating_match%d' % channel]) + self.weights[
                    'sgating_match_bias%d' % channel]))

        def channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(
                    tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(
                    tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings, score

        def item_channel_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(
                    tf.multiply(self.weights['attention_i'], tf.matmul(embedding, self.weights['attention_mat_i'])), 1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(
                    tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings, score

        def match_attention(*channel_embeddings):
            weights = []
            for embedding in channel_embeddings:
                weights.append(tf.reduce_sum(
                    tf.multiply(self.weights['attention_match_channel'],
                                tf.matmul(embedding, self.weights['attention_match_channel_mat'])), 1))
            score = tf.nn.softmax(tf.transpose(weights))
            mixed_embeddings = 0
            for i in range(len(weights)):
                mixed_embeddings += tf.transpose(
                    tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
            return mixed_embeddings, score

        # initialize adjacency matrices
        # with tf.device('/gpu:0'):
        H_s = M_matrices[0]
        H_s = self.adj_to_sparse_tensor(H_s)
        H_j = M_matrices[1]
        H_j = self.adj_to_sparse_tensor(H_j)
        H_p = M_matrices[2]
        H_p = self.adj_to_sparse_tensor(H_p)
        R = self.buildJointAdjacency()

        # H_i = M_matrices[3]
        # H_i = self.adj_to_sparse_tensor(H_i)
        # self-gating
        if 'Social' in self.channels:
            user_embeddings_c1 = self_gating(self.user_embeddings, 1)
        if 'Joint' in self.channels:
            user_embeddings_c2 = self_gating(self.user_embeddings, 2)
        if 'Purchase' in self.channels:
            user_embeddings_c3 = self_gating(self.user_embeddings, 3)
        # simple_user_embeddings = self_gating(self.user_embeddings, 4)
        simple_user_embeddings = self_gating(self.user_embeddings, len(self.channels) + 1)
        if 'Social' in self.channels:
            all_embeddings_c1 = [user_embeddings_c1]
            all_embeddings_c1_match_part = []
        if 'Joint' in self.channels:
            all_embeddings_c2 = [user_embeddings_c2]
            all_embeddings_c2_match_part = []
        if 'Purchase' in self.channels:
            all_embeddings_c3 = [user_embeddings_c3]
            all_embeddings_c3_match_part = []
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings
        # item_embeddings_c1 = self_gating(self.item_embeddings, 5)
        all_embeddings_i = [item_embeddings]
        self.ss_loss = 0
        self.match_ss_loss = 0
        # multi-channel convolution
        for k in range(self.n_layers):
            # Channel S
            if 'Social' in self.channels:
                user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s, user_embeddings_c1)
                # item_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_i, item_embeddings_c1)
                # user_embeddings_c1 = tf.nn.dropout(user_embeddings_c1, self.drop_rate)
                # self.watch['user_embeddings_c1_n_layers_{}'.format(k)] = user_embeddings_c1
                norm_embeddings_c1 = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            # norm_embeddings_c1 = tf.layers.batch_normalization(user_embeddings_c1,
            #                                                    training=self.tf_is_training)
            # norm_embeddings_c1 = tf.nn.relu(norm_embeddings_c1)
            # self.watch['norm_embeddings_c1_n_layers_{}'.format(k)] = norm_embeddings_c1
            # all_embeddings_c1 += [norm_embeddings_c1]  # 换到后面
            # Channel J
            if 'Joint' in self.channels:
                user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
                # user_embeddings_c2 = tf.nn.dropout(user_embeddings_c2, self.drop_rate)
                # self.watch['user_embeddings_c2_n_layers_{}'.format(k)] = user_embeddings_c2
                norm_embeddings_c2 = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            # norm_embeddings_c2 = tf.layers.batch_normalization(user_embeddings_c2,
            #                                                   training=self.tf_is_training)
            # norm_embeddings_c2 = tf.nn.relu(norm_embeddings_c2)
            # self.watch['norm_embeddings_c2_n_layers_{}'.format(k)] = norm_embeddings_c2
            # all_embeddings_c2 += [norm_embeddings_c2]  # 换到后面
            # Channel P
            if 'Purchase' in self.channels:
                user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
                # user_embeddings_c3 = tf.nn.dropout(user_embeddings_c3, self.drop_rate)
                # self.watch['user_embeddings_c3_n_layers_{}'.format(k)] = user_embeddings_c3
                norm_embeddings_c3 = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            # norm_embeddings_c3 = tf.layers.batch_normalization(user_embeddings_c3,
            #                                                    training=self.tf_is_training)
            # norm_embeddings_c3 = tf.nn.relu(norm_embeddings_c3)
            # self.watch['norm_embeddings_c3_n_layers_{}'.format(k)] = norm_embeddings_c3
            # all_embeddings_c3 += [norm_embeddings_c3]  # # 换到后面
            # item convolution
            # todo 修改这里
            # todo 1. 获取各个channel 下的子图表示
            # if self.with_match:
            if 'Social' in self.channels and len(self.channels) > 1:
                topic_rep_c1 = self.get_one_hop_topic_graph_rep(norm_embeddings_c1, H_s)
            # self.watch['topic_rep_c1_n_layers_{}'.format(k)] = topic_rep_c1
            # topic_rep_c1 = tf.math.l2_normalize(topic_rep_c1, axis=1)
            if 'Joint' in self.channels and len(self.channels) > 1:
                topic_rep_c2 = self.get_one_hop_topic_graph_rep(norm_embeddings_c2, H_j)
            # self.watch['topic_rep_c2_n_layers_{}'.format(k)] = topic_rep_c2
            # topic_rep_c2 = tf.math.l2_normalize(topic_rep_c2, axis=1)
            if 'Purchase' in self.channels and len(self.channels) > 1:
                topic_rep_c3 = self.get_one_hop_topic_graph_rep(norm_embeddings_c3, H_p)
            # self.watch['topic_rep_c3_n_layers_{}'.format(k)] = topic_rep_c3
            # topic_rep_c3 = tf.math.l2_normalize(topic_rep_c3, axis=1)
            # todo  2. 计算本channel 下的节点与其他channel 下对应节点的余弦相似度
            # todo 3. 计算其他channel 对本节点的表示
            if 'Social' in self.channels and 'Joint' in self.channels:
                cos12 = self.cosine(topic_rep_c1, topic_rep_c2)
            # self.watch['cos12_n_layers_{}'.format(k)] = cos12
            if 'Social' in self.channels and 'Purchase' in self.channels:
                cos13 = self.cosine(topic_rep_c1, topic_rep_c3)
                # cos13=tf.expand_dims(cos13,dim=-1)
                # cos13 = tf.squeeze(cos13)
            # self.watch['cos13_n_layers_{}'.format(k)] = cos13
            if 'Purchase' in self.channels and 'Joint' in self.channels:
                cos23 = self.cosine(topic_rep_c2, topic_rep_c3)
            # self.watch['cos23_n_layers_{}'.format(k)] = cos23
            if 'Joint' in self.channels and len(self.channels) > 1:
                topic_rep_c2 = tf.expand_dims(topic_rep_c2 * topic_rep_c2, -1)
            # self.watch['topic_rep_c22_n_layers_{}'.format(k)] = topic_rep_c22
            if 'Purchase' in self.channels and len(self.channels) > 1:
                topic_rep_c3 = tf.expand_dims(topic_rep_c3 * topic_rep_c3, -1)
            # self.watch['topic_rep_c33_n_layers_{}'.format(k)] = topic_rep_c33
            if 'Social' in self.channels and len(self.channels) > 1:
                topic_rep_c1 = tf.expand_dims(topic_rep_c1 * topic_rep_c1, -1)
            # self.watch['topic_rep_c11_n_layers_{}'.format(k)] = topic_rep_c11
            if 'Social' in self.channels and 'Joint' in self.channels and 'Purchase' in self.channels:
                topic_rep_c1_match = tf.einsum('ijk, i->ijk', topic_rep_c3, cos13) + tf.einsum('ijk, i->ijk',
                                                                                               topic_rep_c2, cos12)
                topic_rep_c1_match = tf.squeeze(topic_rep_c1_match)
            elif 'Social' in self.channels and 'Joint' in self.channels:
                topic_rep_c1_match = tf.einsum('ijk, i->ijk', topic_rep_c2, cos12)
                topic_rep_c1_match = tf.squeeze(topic_rep_c1_match)
            elif 'Social' in self.channels and 'Purchase' in self.channels:
                topic_rep_c1_match = tf.einsum('ijk, i->ijk', topic_rep_c3, cos13)
                topic_rep_c1_match = tf.squeeze(topic_rep_c1_match)

            # self.watch['topic_rep_c1_match--1_n_layers_{}'.format(k)] = topic_rep_c1_match
            # self.watch['topic_rep_c1_match--2_n_layers_{}'.format(k)] = topic_rep_c1_match
            # todo nan问题出在这
            # todo 这一步除法有问题
            # topic_rep_c1_match = tf.div(topic_rep_c1_match, tf.expand_dims(tf.add(cos13, cos12), -1))
            # self.watch['topic_rep_c1_match_n_layers_{}'.format(k)] = topic_rep_c1_match
            if 'Social' in self.channels and 'Joint' in self.channels and 'Purchase' in self.channels:
                topic_rep_c2_match = tf.einsum('ijk, i->ijk', topic_rep_c3, cos23) + tf.einsum('ijk, i->ijk',
                                                                                               topic_rep_c1, cos12)
                topic_rep_c2_match = tf.squeeze(topic_rep_c2_match)
            elif 'Social' in self.channels and 'Joint' in self.channels:
                topic_rep_c2_match = tf.einsum('ijk, i->ijk', topic_rep_c1, cos12)
                topic_rep_c2_match = tf.squeeze(topic_rep_c2_match)
            elif 'Joint' in self.channels and 'Purchase' in self.channels:
                topic_rep_c2_match = tf.einsum('ijk, i->ijk', topic_rep_c3, cos23)
                topic_rep_c2_match = tf.squeeze(topic_rep_c2_match)

            # topic_rep_c2_match = tf.einsum('ijk, i->ijk', topic_rep_c3, cos23) + tf.einsum('ijk, i->ijk',
            #                                                                                topic_rep_c1, cos12)
            # topic_rep_c2_match = tf.squeeze(topic_rep_c2_match)
            # topic_rep_c2_match = tf.div(topic_rep_c2_match, tf.expand_dims(tf.add(cos23, cos12), -1))
            # self.watch['topic_rep_c2_match_n_layers_{}'.format(k)] = topic_rep_c2_match

            if 'Social' in self.channels and 'Joint' in self.channels and 'Purchase' in self.channels:
                topic_rep_c3_match = tf.einsum('ijk, i->ijk', topic_rep_c2, cos23) + tf.einsum('ijk, i->ijk',
                                                                                               topic_rep_c1, cos13)
                topic_rep_c3_match = tf.squeeze(topic_rep_c3_match)
            elif 'Social' in self.channels and 'Purchase' in self.channels:
                topic_rep_c3_match = tf.einsum('ijk, i->ijk', topic_rep_c1, cos13)
                topic_rep_c3_match = tf.squeeze(topic_rep_c3_match)
            elif 'Joint' in self.channels and 'Purchase' in self.channels:
                topic_rep_c3_match = tf.einsum('ijk, i->ijk', topic_rep_c2, cos23)
                topic_rep_c3_match = tf.squeeze(topic_rep_c3_match)

            # topic_rep_c3_match = tf.div(topic_rep_c3_match, tf.expand_dims(tf.add(cos23, cos13), -1))
            # self.watch['topic_rep_c3_match_n_layers_{}'.format(k)] = topic_rep_c3_match
            # todo 4. 本 channel 下节点表示与步骤3的融合表示 fm
            # c1_user_match_final = self.fm(norm_embeddings_c1, topic_rep_c1_match)
            if 'Social' in self.channels and len(self.channels) > 1:
                c1_user_match_final = self.fm(user_embeddings_c1, topic_rep_c1_match, self.with_mp_cosine,
                                              self.with_cosine)
                norm_c1_user_match_final = tf.math.l2_normalize(c1_user_match_final, axis=1)
                all_embeddings_c1_match_part += [norm_c1_user_match_final]
            # c1_user_match_final = tf.nn.dropout(c1_user_match_final, self.drop_rate)
            # self.watch['c1_user_match_final_match_n_layers_{}'.format(k)] = c1_user_match_final
            # c2_user_match_final = self.fm(norm_embeddings_c2, topic_rep_c2_match)
            if 'Joint' in self.channels and len(self.channels) > 1:
                c2_user_match_final = self.fm(user_embeddings_c2, topic_rep_c2_match, self.with_mp_cosine,
                                              self.with_cosine)
                norm_c2_user_match_final = tf.math.l2_normalize(c2_user_match_final, axis=1)
                all_embeddings_c2_match_part += [norm_c2_user_match_final]
            # c2_user_match_final = tf.nn.dropout(c2_user_match_final, self.drop_rate)
            # self.watch['c2_user_match_final_match_n_layers_{}'.format(k)] = c2_user_match_final
            # c3_user_match_final = self.fm(norm_embeddings_c3, topic_rep_c3_match)
            if 'Purchase' in self.channels and len(self.channels) > 1:
                c3_user_match_final = self.fm(user_embeddings_c3, topic_rep_c3_match, self.with_mp_cosine,
                                              self.with_cosine)
                norm_c3_user_match_final = tf.math.l2_normalize(c3_user_match_final, axis=1)
                all_embeddings_c3_match_part += [norm_c3_user_match_final]
            # c3_user_match_final = tf.nn.dropout(c3_user_match_final, self.drop_rate)
            # self.watch['c3_user_match_final_match_n_layers_{}'.format(k)] = c3_user_match_final

            # print 'topic_rep_c12_match={}'.format(topic_rep_c12_match.eval())

            # todo 5.  GNN#2 加入
            # c1_user_match_final = tf.sparse_tensor_dense_matmul(H_s, c1_user_match_final)
            # c1_user_match_final = tf.nn.dropout(c1_user_match_final, self.drop_rate)
            # c2_user_match_final = tf.sparse_tensor_dense_matmul(H_j, c2_user_match_final)
            # c2_user_match_final = tf.nn.dropout(c2_user_match_final, self.drop_rate)
            # c3_user_match_final = tf.sparse_tensor_dense_matmul(H_p, c3_user_match_final)
            # c3_user_match_final = tf.nn.dropout(c3_user_match_final, self.drop_rate)

            # todo 6. 方案1：本 channel 下节点表示拼接最为最终本 channel 下的表示，这样下一步不变
            # norm_c1_user_match_final = tf.math.l2_normalize(c1_user_match_final, axis=1)
            # norm_c1_user_match_final = tf.layers.batch_normalization(c1_user_match_final,
            #                                                    training=self.tf_is_training)
            # norm_c1_user_match_final = tf.nn.relu(norm_c1_user_match_final)
            # c1 = norm_embeddings_c1 * (1 - self.match_ratio) + norm_c1_user_match_final * self.match_ratio
            # attention 版本
            if self.with_match:
                if 'Social' in self.channels and len(self.channels) > 1:
                    c1, s = match_attention(norm_embeddings_c1, norm_c1_user_match_final)
                elif 'Social' in self.channels:
                    c1 = norm_embeddings_c1
                if 'Joint' in self.channels and len(self.channels) > 1:
                    c2, s2 = match_attention(norm_embeddings_c2, norm_c2_user_match_final)
                elif 'Joint' in self.channels:
                    c2 = norm_embeddings_c2
                if 'Purchase' in self.channels and len(self.channels) > 1:
                    c3, s3 = match_attention(norm_embeddings_c3, norm_c3_user_match_final)
                elif 'Purchase' in self.channels:
                    c3 = norm_embeddings_c3
                # c1 = norm_c1_user_match_final
                # c2 = norm_c2_user_match_final
                # c3 = norm_c3_user_match_final
            else:
                c1 = norm_embeddings_c1
                c2 = norm_embeddings_c2
                c3 = norm_embeddings_c3
            # norm_embeddings_c1 * (1 - self.match_ratio) + norm_c1_user_match_final * self.match_ratio
            # all_embeddings_c1 += [c1 / 2]  # 182行
            if 'Social' in self.channels:
                all_embeddings_c1 += [c1]  # 182行
            # norm_c2_user_match_final = tf.layers.batch_normalization(c2_user_match_final,
            #                                                          training=self.tf_is_training)
            # norm_c2_user_match_final = tf.nn.relu(norm_c2_user_match_final)
            # c2 = norm_embeddings_c2 + norm_c2_user_match_final
            # c2 = norm_embeddings_c2 * (1 - self.match_ratio) + norm_c2_user_match_final * self.match_ratio
            # all_embeddings_c2 += [c2 / 2]  # 186行
            if 'Joint' in self.channels:
                all_embeddings_c2 += [c2]  # 186行
            # norm_c3_user_match_final = tf.layers.batch_normalization(c3_user_match_final,
            #                                                          training=self.tf_is_training)
            # norm_c3_user_match_final = tf.nn.relu(norm_c3_user_match_final)
            # c3 = norm_embeddings_c3 + norm_c3_user_match_final
            # c3 = norm_embeddings_c3 * (1 - self.match_ratio) + norm_c3_user_match_final * self.match_ratio
            # all_embeddings_c3 += [c3 / 2]  # 190行
            if 'Purchase' in self.channels:
                all_embeddings_c3 += [c3]  # 190行
            # tf.Print(a, ['a_value: ', a])
            # mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[
            #                       0] + simple_user_embeddings / 2
            # mixed_embedding = \
            #     channel_attention(user_embeddings_c1 * (1 - self.match_ratio) + c1_user_match_final * self.match_ratio,
            #                       user_embeddings_c2 * (1 - self.match_ratio) + c2_user_match_final * self.match_ratio,
            #                       user_embeddings_c3 * (1 - self.match_ratio) + c3_user_match_final * self.match_ratio)[
            #         0] + simple_user_embeddings / 2

            # attention 版本
            if self.with_match:
                if 'Social' in self.channels and len(self.channels) > 1:
                    ccc1, _ = match_attention(user_embeddings_c1, c1_user_match_final)
                elif 'Social' in self.channels:
                    ccc1 = user_embeddings_c1
                if 'Joint' in self.channels and len(self.channels) > 1:
                    ccc2, _ = match_attention(user_embeddings_c2, c2_user_match_final)
                elif 'Joint' in self.channels:
                    ccc2 = user_embeddings_c2
                if 'Purchase' in self.channels and len(self.channels) > 1:
                    ccc3, _ = match_attention(user_embeddings_c3, c3_user_match_final)
                elif 'Purchase' in self.channels:
                    ccc3 = user_embeddings_c3
                # ccc1, _ = match_attention(user_embeddings_c1, c1_user_match_final)
                # ccc2, _ = match_attention(user_embeddings_c2, c2_user_match_final)
                # ccc3, _ = match_attention(user_embeddings_c3, c3_user_match_final)
                if 'Social' in self.channels and 'Joint' in self.channels and 'Purchase' in self.channels:
                    mixed_embedding = channel_attention(ccc1, ccc2, ccc3)[0] + simple_user_embeddings / 2
                elif 'Social' in self.channels and 'Joint' in self.channels:
                    mixed_embedding = channel_attention(ccc1, ccc2)[0] + simple_user_embeddings / 2
                elif 'Social' in self.channels and 'Purchase' in self.channels:
                    mixed_embedding = channel_attention(ccc1, ccc3)[0] + simple_user_embeddings / 2
                elif 'Joint' in self.channels and 'Purchase' in self.channels:
                    mixed_embedding = channel_attention(ccc2, ccc3)[0] + simple_user_embeddings / 2
                elif 'Social' in self.channels:
                    mixed_embedding = ccc1 + simple_user_embeddings / 2
                elif 'Joint' in self.channels:
                    mixed_embedding = ccc2 + simple_user_embeddings / 2
                elif 'Purchase' in self.channels:
                    mixed_embedding = ccc3 + simple_user_embeddings / 2
                    # mixed_embedding = channel_attention(c1_user_match_final, c2_user_match_final, c3_user_match_final)[
                #                       0] + simple_user_embeddings / 2
            else:
                # ccc1, _ = user_embeddings_c1
                # ccc2, _ = user_embeddings_c2
                # ccc3, _ = user_embeddings_c3
                mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[
                                      0] + simple_user_embeddings / 2
            # mixed_embedding = tf.layers.dropout(mixed_embedding, rate=self.drop_rate, training=self.tf_is_training)
            # mixed_embedding = tf.layers.batch_normalization(mixed_embedding, training=self.tf_is_training)
            # mixed_embedding = tf.nn.relu(mixed_embedding)

            # mixed_embedding = tf.concat(
            #     【user_embeddings_c1 * (1 - self.match_ratio) + c1_user_match_final * self.match_ratio,
            #     user_embeddings_c2 * (1 - self.match_ratio) + c2_user_match_final * self.match_ratio,
            #     user_embeddings_c3 * (1 - self.match_ratio) + c3_user_match_final * self.match_ratio,
            #     simple_user_embeddings / 2)
            # channel_attention(user_embeddings_c1 * (1 - self.match_ratio) + c1_user_match_final * self.match_ratio,
            #                   user_embeddings_c2 * (1 - self.match_ratio) + c2_user_match_final * self.match_ratio,
            #                   user_embeddings_c3 * (1 - self.match_ratio) + c3_user_match_final * self.match_ratio)[
            #     0] + simple_user_embeddings / 2
            # mixed_embedding = channel_attention((user_embeddings_c1 + norm_c1_user_match_final) / 2,
            #                                     (user_embeddings_c2 + norm_c2_user_match_final) / 2,
            #                                     (user_embeddings_c3 + norm_c3_user_match_final) / 2)[
            #                       0] + simple_user_embeddings / 2
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            # new_item_embeddings, a = item_channel_attention(new_item_embeddings, item_embeddings_c1)
            # new_item_embeddings = tf.nn.dropout(new_item_embeddings, self.drop_rate)
            norm_embeddings_item = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings_item]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            # simple_user_embeddings = tf.nn.dropout(simple_user_embeddings, self.drop_rate)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        # averaging the channel-specific embeddings
        if 'Social' in self.channels:
            user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        # user_embeddings_c1 = tf.nn.dropout(user_embeddings_c1, self.drop_rate)
        if 'Social' in self.channels and len(self.channels) > 1:
            user_embeddings_c1_match = tf.reduce_sum(all_embeddings_c1_match_part, axis=0)
        if 'Joint' in self.channels:
            user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        if 'Joint' in self.channels and len(self.channels) > 1:
            user_embeddings_c2_match = tf.reduce_sum(all_embeddings_c2_match_part, axis=0)
        # user_embeddings_c2 = tf.nn.dropout(user_embeddings_c2, self.drop_rate)
        if 'Purchase' in self.channels:
            user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        if 'Purchase' in self.channels and len(self.channels) > 1:
            user_embeddings_c3_match = tf.reduce_sum(all_embeddings_c3_match_part, axis=0)
        # user_embeddings_c3 = tf.nn.dropout(user_embeddings_c3, self.drop_rate)

        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        # simple_user_embeddings = tf.nn.dropout(simple_user_embeddings, self.drop_rate)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
        # item_embeddings = tf.nn.dropout(item_embeddings, self.drop_rate)
        # aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        if 'Social' in self.channels and 'Joint' in self.channels and 'Purchase' in self.channels:
            self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1,
                                                                                 user_embeddings_c2,
                                                                                 user_embeddings_c3)
        elif 'Social' in self.channels and 'Joint' in self.channels:
            self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1,
                                                                                 user_embeddings_c2)
        elif 'Social' in self.channels and 'Purchase' in self.channels:
            self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1,
                                                                                 user_embeddings_c3)
        elif 'Joint' in self.channels and 'Purchase' in self.channels:
            self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c2,
                                                                                 user_embeddings_c3)
        elif 'Social' in self.channels:
            self.final_user_embeddings = user_embeddings_c1
        elif 'Joint' in self.channels:
            self.final_user_embeddings = user_embeddings_c2
        elif 'Purchase' in self.channels:
            self.final_user_embeddings = user_embeddings_c3

        # self.final_user_embeddings += simple_user_embeddings/2  # for yelp and douban, 1/2 simple user embedding is better
        self.final_user_embeddings += simple_user_embeddings  # for yelp and douban, 1/2 simple user embedding is better
        # self.final_user_embeddings = tf.layers.dropout(self.final_user_embeddings, rate=self.drop_rate,
        #                                                training=self.tf_is_training)
        # self.final_user_embeddings = tf.layers.batch_normalization( self.final_user_embeddings, training=self.tf_is_training)
        # self.final_user_embeddings = tf.nn.relu( self.final_user_embeddings)
        # self.final_user_embeddings = tf.nn.relu(self.final_user_embeddings)

        # create self-supervised loss
        # 利用final_user_embeddings 在各个模态关系上进行GNN 再进行SSL
        # user_embeddings_c1_ssl=tf.sparse_tensor_dense_matmul(H_s, self_supervised_gating(self.final_user_embeddings, 1))
        # user_embeddings_c2_ssl=tf.sparse_tensor_dense_matmul(H_j, self_supervised_gating(self.final_user_embeddings, 2))
        # user_embeddings_c3_ssl=tf.sparse_tensor_dense_matmul(H_p, self_supervised_gating(self.final_user_embeddings, 3))

        # self.ss_loss += self.modal_ssl(user_embeddings_c1_ssl,
        #                                user_embeddings_c2_ssl)
        # self.ss_loss += self.modal_ssl(user_embeddings_c2_ssl,
        #                                user_embeddings_c3_ssl)
        # self.ss_loss += self.modal_ssl(user_embeddings_c1_ssl,
        #                                user_embeddings_c3_ssl)
        # todo DHCF里面的高阶可以加进去试试
        # todo ssl 里面还没加 dropout
        # ssl gating 模态
        if self.with_ssl:
            if 'Social' in self.channels:
                self.ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating(self.final_user_embeddings, 1), H_s)
            if 'Joint' in self.channels:
                self.ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating(self.final_user_embeddings, 2), H_j)
            if 'Purchase' in self.channels:
                self.ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating(self.final_user_embeddings, 3), H_p)
            # self.ss_loss += self.hierarchical_mutual_information_maximization(
            #     self_supervised_gating(self.final_item_embeddings, 4), H_i)

            # self.ss_loss += self.modal_ssl(user_embeddings_c1, user_embeddings_c2)
            # self.ss_loss += self.modal_ssl(user_embeddings_c1, user_embeddings_c3)
            # self.ss_loss += self.modal_ssl(user_embeddings_c2, user_embeddings_c3)

            # self.ss_loss += self.modal_ssl(self_supervised_gating(user_embeddings_c1, 1),
            #                                self_supervised_gating(user_embeddings_c2, 2))
            # self.ss_loss += self.modal_ssl(self_supervised_gating(user_embeddings_c2, 2),
            #                                self_supervised_gating(user_embeddings_c3, 3))
            # self.ss_loss += self.modal_ssl(self_supervised_gating(user_embeddings_c1, 1),
            #                                self_supervised_gating(user_embeddings_c3, 3))
        if self.with_split_match_ssl:
            # all_embeddings_c1_match_part=
            # self.ss_loss += self.modal_ssl(self_supervised_gating_for_match(user_embeddings_c1_match, 1),
            #                                self_supervised_gating_for_match(user_embeddings_c2_match, 2))
            # self.ss_loss += self.modal_ssl(self_supervised_gating_for_match(user_embeddings_c2_match, 2),
            #                                self_supervised_gating_for_match(user_embeddings_c3_match, 3))
            # self.ss_loss += self.modal_ssl(self_supervised_gating_for_match(user_embeddings_c1_match, 1),
            #                                self_supervised_gating_for_match(user_embeddings_c3_match, 3))
            #
            # self.match_ss_loss += self.hierarchical_modal_ssl(
            #     self_supervised_gating_for_match(user_embeddings_c1_match, 1),
            #     self_supervised_gating_for_match(user_embeddings_c2_match, 2),
            #     H_s, H_j)
            # self.match_ss_loss += self.hierarchical_modal_ssl(
            #     self_supervised_gating_for_match(user_embeddings_c2_match, 2),
            #     self_supervised_gating_for_match(user_embeddings_c3_match, 3),
            #     H_j, H_p)
            # self.match_ss_loss += self.hierarchical_modal_ssl(
            #     self_supervised_gating_for_match(user_embeddings_c1_match, 1),
            #     self_supervised_gating_for_match(user_embeddings_c3_match, 3),
            #     H_s, H_p)
            if 'Social' in self.channels and 'Joint' in self.channels:
                self.match_ss_loss += self.modal_ssl_INfoNce(
                    self_supervised_gating_for_match(user_embeddings_c1_match, 1),
                    self_supervised_gating_for_match(user_embeddings_c2_match, 2))
            if 'Joint' in self.channels and 'Purchase' in self.channels:
                self.match_ss_loss += self.modal_ssl_INfoNce(
                    self_supervised_gating_for_match(user_embeddings_c2_match, 2),
                    self_supervised_gating_for_match(user_embeddings_c3_match, 3))
            if 'Social' in self.channels and 'Purchase' in self.channels:
                self.match_ss_loss += self.modal_ssl_INfoNce(
                    self_supervised_gating_for_match(user_embeddings_c1_match, 1),
                    self_supervised_gating_for_match(user_embeddings_c3_match, 3))
            if 'Social' in self.channels and len(self.channels) > 1:
                self.match_ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating_for_match(user_embeddings_c1_match, 1), H_s)
            if 'Joint' in self.channels and len(self.channels) > 1:
                self.match_ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating_for_match(user_embeddings_c2_match, 2), H_j)
            if 'Purchase' in self.channels and len(self.channels) > 1:
                self.match_ss_loss += self.hierarchical_mutual_information_maximization(
                    self_supervised_gating_for_match(user_embeddings_c3_match, 3), H_p)

        # embedding look-up
        self.neg_item_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
        # self.u_embedding_c1_match_part = tf.nn.embedding_lookup(all_embeddings_c1_match_part, self.u_idx)
        # self.u_embedding_c2_match_part = tf.nn.embedding_lookup(all_embeddings_c2_match_part, self.u_idx)
        # self.u_embedding_c3_match_part = tf.nn.embedding_lookup(all_embeddings_c3_match_part, self.u_idx)
        # self.u_all_embeddings_c1 = tf.nn.embedding_lookup(all_embeddings_c1, self.u_idx)
        # self.u_all_embeddings_c2 = tf.nn.embedding_lookup(all_embeddings_c2, self.u_idx)
        # self.u_all_embeddings_c3 = tf.nn.embedding_lookup(all_embeddings_c3, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)

    def hierarchical_mutual_information_maximization(self, em, adj):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(
                tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding,
                                            tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding

        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)

        user_embeddings = em
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        # user_embeddings1 = row_shuffle(user_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        # edge_embeddings1 = row_column_shuffle(edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        #  为什么是neg1 - neg2？换成加号试试 解决：是相对值，这么理解
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
        # Global MIM
        graph = tf.reduce_mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def modal_ssl(self, em1, em2):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(
                tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding,
                                            tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding

        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)

        # user_embeddings = em1
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        # edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        # Local MIM
        pos = score(em1, em2)
        neg1 = score(row_shuffle(em1), em2)
        neg2 = score(row_column_shuffle(em2), em1)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
        # Global MIM
        # graph = tf.reduce_mean(edge_embeddings, 0)
        # pos = score(edge_embeddings, graph)
        # neg1 = score(row_column_shuffle(edge_embeddings), graph)
        # global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)))
        # return global_loss + local_loss
        return local_loss

    def modal_ssl_INfoNce(self, em1, em2):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(
                tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding,
                                            tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding

        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)

        # user_embeddings = em1
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        # edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        # Local MIM
        em1 = tf.nn.embedding_lookup(em1, self.u_idx)
        em2 = tf.nn.embedding_lookup(em2, self.u_idx)
        em1 = tf.nn.l2_normalize(em1, 1)
        em2 = tf.nn.l2_normalize(em2, 1)
        # pos_score_user = tf.reduce_sum(tf.multiply(em1, em2), axis=1)
        # ttl_score_user = tf.matmul(em1, em1, transpose_a=False, transpose_b=True)
        # pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        # pos_score_user = tf.exp(pos_score_user / 0.5)
        # ttl_score_user = tf.reduce_sum(tf.exp(tf.matmul(em1, em1, transpose_a=False, transpose_b=True) / 0.5), axis=1)
        ssl_loss_user = -tf.reduce_sum(tf.log(
            tf.exp(tf.reduce_sum(tf.multiply(em1, em2), axis=1) / 0.5) / tf.reduce_sum(
                tf.exp(tf.matmul(em1, em1, transpose_a=False, transpose_b=True) / 0.5), axis=1)))

        # pos = score(em1, em2)
        # neg1 = score(row_shuffle(em1), em2)
        # neg2 = score(row_column_shuffle(em2), em1)
        # local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
        # Global MIM
        # graph = tf.reduce_mean(edge_embeddings, 0)
        # pos = score(edge_embeddings, graph)
        # neg1 = score(row_column_shuffle(edge_embeddings), graph)
        # global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)))
        # return global_loss + local_loss
        return ssl_loss_user

    def hierarchical_modal_ssl(self, em1, em2, adj1, adj2):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(
                tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding,
                                            tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding

        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)

        # user_embeddings = em1
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings1 = tf.sparse_tensor_dense_matmul(adj1, em1)
        edge_embeddings2 = tf.sparse_tensor_dense_matmul(adj2, em2)
        # Local MIM
        pos = score(em1, em2)
        neg1 = score(row_shuffle(em1), em2)
        neg2 = score(row_column_shuffle(em2), em1)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
        # Global MIM
        graph1 = tf.reduce_mean(edge_embeddings1, 0)
        graph2 = tf.reduce_mean(edge_embeddings2, 0)
        pos1 = score(edge_embeddings1, edge_embeddings2)
        pos2 = score(tf.expand_dims(graph1, -1), tf.expand_dims(graph2, -1))
        neg11 = score(row_shuffle(edge_embeddings1), edge_embeddings2)
        neg12 = score(row_column_shuffle(edge_embeddings1), edge_embeddings2)
        neg21 = score(row_shuffle(tf.expand_dims(graph1, -1)), tf.expand_dims(graph2, -1))
        # neg22 = score(row_column_shuffle(graph1), graph2)
        global_loss = tf.reduce_sum(
            -tf.log(tf.sigmoid(pos1 - neg11)) - tf.log(tf.sigmoid(neg11 - neg12))) + tf.reduce_sum(
            -tf.log(tf.sigmoid(pos2 - neg21)))
        return global_loss + local_loss
        # return local_loss

    def get_one_hop_topic_graph_rep(self, em, adj):
        user_embeddings = em
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        return edge_embeddings

    def fm(self, em1, em2, with_mp_cosine=False, with_cosine=True):
        input_shape = tf.shape(em1)
        batch_size = input_shape[0]
        # node_rep_dim = int(input_shape[1])
        # node_rep_dim = input_shape[2]
        # [batch_size, single_graph_1_nodes_size, match_dim]
        (max_attentive_rep, _) = self.multi_perspective_match(self.node_dim, em1, em2, with_mp_cosine, with_cosine,
                                                              scope_name='mp-match-max-att')
        # em1 = tf.expand_dims(em1, -1)  # N*d*1
        # em2 = tf.expand_dims(em2, -1)  # N*d*1
        # W = tf.transpose(self.weights['match_weight'])  # d*L
        #
        #
        # edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        max_attentive_rep = tf.squeeze(max_attentive_rep)
        return max_attentive_rep

    def multi_perspective_match(self, feature_dim, rep_1, rep_2, with_mp_cosine=None, with_cosine=None,
                                scope_name='mp-match',
                                reuse=tf.compat.v1.AUTO_REUSE):
        '''
            :param repres1: [batch_size, len, feature_dim]
            :param repres2: [batch_size, len, feature_dim]
            :return:
        '''
        rep_1 = tf.expand_dims(rep_1, dim=1)
        rep_2 = tf.expand_dims(rep_2, dim=1)
        input_shape = tf.shape(rep_1)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        matching_result = []
        with tf.variable_scope(scope_name, reuse=reuse):
            match_dim = 0
            if with_cosine:
                cosine_value = layer_utils.cosine_distance(rep_1, rep_2, cosine_norm=False)
                cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
                matching_result.append(cosine_value)
                match_dim += 1

            if with_mp_cosine:
                # mp_cosine_params = tf.get_variable("mp_cosine", shape=[self.perspect, feature_dim],
                #                                    dtype=tf.float32)
                mp_cosine_params = self.weights['match_weight']
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                rep_1_flat = tf.expand_dims(rep_1, axis=2)
                rep_2_flat = tf.expand_dims(rep_2, axis=2)
                # te = tf.multiply(rep_1_flat, mp_cosine_params)
                mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(rep_1_flat, mp_cosine_params),
                                                                 rep_2_flat, cosine_norm=False)
                matching_result.append(mp_cosine_matching)
                match_dim += 10

        matching_result = tf.concat(axis=2, values=matching_result)
        return (matching_result, match_dim)

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        # print 'score={}\n'.format(score.eval())
        return score

    def saveModel(self):
        # store the best parameters
        self.bestU, self.bestV, attention_score_best = self.sess.run(
            [self.final_user_embeddings, self.final_item_embeddings, self.attention_score],
            feed_dict={self.tf_is_training: False, self.drop_rate: 0})
        # self.
        # currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # outDir = self.output['-dir'] + 'regU_' + str(self.config[
        #                                                  'regU']) + "/lamda_" + str(
        #     self.config['lamda']) + '/ss_rate_' + str(self.config[
        #                                                   'ss_rate']) + '/ss_model_rate' + str(self.config[
        #                                                                                            'ss_model_rate']) + '/measure/'
        # if not os.path.exists(outDir):
        #     os.makedirs(outDir)
        # attention_file_name = self.config[
        #                           'recommender'] + '@' + currentTime + '-channel_attention' + self.foldInfo + '.txt'
        # with open(outDir + attention_file_name, 'w') as f:
        #     np.savetxt(f, attention_score_best, delimiter=',', newline='\n')

    def buildModel(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # currentTime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time()))
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        reg_loss = 0
        for key in self.weights:
            reg_loss += self.lamda * tf.nn.l2_loss(self.weights[key])
        rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (
                tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        total_loss = rec_loss + reg_loss + self.ss_rate * self.ss_loss + self.ss_model_rate * self.match_ss_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train_op = opt.minimize(total_loss)
        # init = tf.global_variables_initializer()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        # Maximum Iteration Setting: LastFM 100 Douban 30 Yelp 30
        for iteration in range(self.maxIter):
            drop_rate = self.max_drop_rate * iteration / self.maxIter
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l1, watch, ss_loss, match_ss_loss = self.sess.run(
                    [train_op, rec_loss, self.watch, self.ss_loss, self.match_ss_loss],
                    feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,
                               self.tf_is_training: True, self.drop_rate: drop_rate})
                # todo attention_score
                # _, l1, watch, ss_loss = self.sess.run(
                #     [train_op, rec_loss, self.watch, self.ss_loss],
                #     feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,
                #                self.tf_is_training: True, self.drop_rate: drop_rate})
                print('[', self.foldInfo, ']', 'training:', iteration + 1, 'batch', n, 'drop_rate:', drop_rate,
                      'rec loss:', l1)  # ,'ss_loss',l2
                # outDir = self.output['-dir'] + 'layer_' + str(self.n_layers) + '/regU_' + str(self.config[
                #                                                                                  'regU']) + "/lamda_" + str(
                #     self.config['lamda']) + '/ss_rate_' + str(self.config[
                #                                                   'ss_rate']) + '/ss_model_rate' + str(self.config[
                #                                                                                            'ss_model_rate']) + '/' + '/training/'
                outDir = self.output['-dir'] + 'ss_rate_' + \
                         str(self.config['ss_rate']) + '/ss_model_rate' + str(
                    self.config['ss_model_rate']) + '/training/'

                # s = '%s\t%s\t%s\n' % (iteration + 1, n, l1)
                # s = '%s\t%s\t%s\t%s\n' % (iteration + 1, n, drop_rate, l1)
                s = '%s\t%s\t%s\t%s\t%s\t%s\n' % (iteration + 1, n, drop_rate, l1, ss_loss, match_ss_loss)
                # s = '%s\t%s\t%s\t%s\t%s\t%s\n' % (iteration + 1, n, drop_rate, l1, ss_loss, 0)
                fileName = self.config[
                               'recommender'] + '@' + currentTime + '-training_progress' + self.foldInfo + '.txt'
                FileIO.writeFileappend(outDir, fileName, s)

            # 测试阶段
            self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings],
                                           feed_dict={self.tf_is_training: False, self.drop_rate: 0})
            # print('[', self.foldInfo, ']', 'test:', iteration + 1, 'test rec loss:', test_loss)  # ,'ss_loss',l2
            # if iteration > self.maxIter - 50:
            # if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == 1:
            if (iteration + 1) % 10 == 0 or iteration == 0 or iteration > (self.maxIter / 3 * 2):
                self.train_ranking_performance(iteration, currentTime)
                self.ranking_performance(iteration, currentTime)
        self.U, self.V = self.bestU, self.bestV

    # def saveLoss(self,filepath,):
    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
