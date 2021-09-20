import texar as tx
import tensorflow as tf
import numpy as np
from preprocess.data_utils import kw_tokenize


class Predictor():
    def __init__(self, config_model, config_data, mode=None):
        self.config = config_model
        self.data_config = config_data
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.build_model()

    def build_model(self):
        self.train_data = tx.data.MultiAlignedData(self.data_config.data_hparams['train'])
        self.valid_data = tx.data.MultiAlignedData(self.data_config.data_hparams['valid'])
        self.test_data = tx.data.MultiAlignedData(self.data_config.data_hparams['test'])
        self.iterator = tx.data.TrainTestDataIterator(train=self.train_data, val=self.valid_data, test=self.test_data)
        self.vocab = self.train_data.vocab(0)
        self.source_encoder = tx.modules.HierarchicalRNNEncoder(hparams=self.config.source_encoder_hparams)
        self.target_encoder = tx.modules.BidirectionalRNNEncoder(hparams=self.config.target_encoder_hparams)
        self.target_kwencoder = tx.modules.BidirectionalRNNEncoder(hparams=self.config.target_kwencoder_hparams)
        self.linear_transform = tx.modules.MLPTransformConnector(self.config._code_len // 2)
        self.linear_matcher = tx.modules.MLPTransformConnector(1)
        self.context_encoder = tx.modules.UnidirectionalRNNEncoder(hparams=self.config.context_encoder_hparams)
        self.predict_layer = tx.modules.MLPTransformConnector(self.data_config._keywords_num)
        self.embedder = tx.modules.WordEmbedder(init_value=self.train_data.embedding_init_value(0).word_vecs)
        self.kw_list = self.vocab.map_tokens_to_ids(tf.convert_to_tensor(self.data_config._keywords_candi))
        self.kw_vocab = tx.data.Vocab(self.data_config._keywords_path)

    def forward_neural(self, context_ids, context_length):
        context_embed = self.embedder(context_ids)
        context_code = self.context_encoder(context_embed, sequence_length=context_length)[1]
        keyword_score = self.predict_layer(context_code)
        return keyword_score

    def predict_keywords(self, batch):
        matching_score = self.forward_neural(batch['context_text_ids'], batch['context_length'])
        keywords_ids = self.kw_vocab.map_tokens_to_ids(batch['keywords_text'])
        kw_labels = tf.map_fn(lambda x: tf.sparse_to_dense(x, [self.kw_vocab.size], 1., 0., False),
                              keywords_ids, dtype=tf.float32, parallel_iterations=True)[:, 4:]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=kw_labels, logits=matching_score)
        loss = tf.reduce_mean(loss)
        kw_ans = tf.arg_max(matching_score, -1)
        acc_label = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (kw_labels, kw_ans), dtype=tf.float32)
        acc = tf.reduce_mean(acc_label)
        kws = tf.nn.top_k(matching_score, k=5)[1]
        kws = tf.reshape(kws,[-1])
        kws = tf.map_fn(lambda x: self.kw_list[x], kws, dtype=tf.int64)
        kws = tf.reshape(kws,[-1, 5])
        return loss, acc, kws

    def train_keywords(self):
        batch = self.iterator.get_next()
        loss, acc, _ = self.predict_keywords(batch)
        op_step = tf.Variable(0, name='op_step')
        train_op = tx.core.get_train_op(loss, global_step=op_step, hparams=self.config.neural_opt_hparams)
        max_val_acc = 0.
        self.saver = tf.train.Saver()
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            for epoch_id in range(self.config._max_epoch):
                self.iterator.switch_to_train_data(sess)
                cur_step = 0
                cnt_acc = []
                while True:
                    try:
                        cur_step += 1
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                        loss_, acc_ = sess.run([train_op, acc], feed_dict=feed)
                        cnt_acc.append(acc_)
                        if cur_step % 200 == 0:
                            print('batch {}, loss={}, acc1={}'.format(cur_step, loss_, np.mean(cnt_acc[-200:])))
                    except tf.errors.OutOfRangeError:
                        break
                self.iterator.switch_to_val_data(sess)
                cnt_acc = []
                while True:
                    try:
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                        acc_ = sess.run(acc, feed_dict=feed)
                        cnt_acc.append(acc_)
                    except tf.errors.OutOfRangeError:
                        mean_acc = np.mean(cnt_acc)
                        if mean_acc > max_val_acc:
                            max_val_acc = mean_acc
                            self.saver.save(sess, self.config._neural_save_path)
                        print('epoch_id {}, valid acc1={}'.format(epoch_id+1, mean_acc))
                        break

    def test_keywords(self):
        batch = self.iterator.get_next()
        loss, acc, kws = self.predict_keywords(batch)
        saver = tf.train.Saver()
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            saver.restore(sess, self.config._neural_save_path)
            self.iterator.switch_to_test_data(sess)
            cnt_acc, cnt_rec1, cnt_rec3, cnt_rec5 = [], [], [], []
            while True:
                try:
                    feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
                    acc_, kw_ans, kw_labels = sess.run([acc, kws, batch['keywords_text_ids']], feed_dict=feed)
                    cnt_acc.append(acc_)
                    rec = [0,0,0,0,0]
                    sum_kws = 0
                    for i in range(len(kw_ans)):
                        sum_kws += sum(kw_labels[i] > 3)
                        for j in range(5):
                            if kw_ans[i][j] in kw_labels[i]:
                                for k in range(j, 5):
                                    rec[k] += 1
                    cnt_rec1.append(rec[0]/sum_kws)
                    cnt_rec3.append(rec[2]/sum_kws)
                    cnt_rec5.append(rec[4]/sum_kws)

                except tf.errors.OutOfRangeError:
                    print('test_kw acc@1={:.4f}, rec@1={:.4f}, rec@3={:.4f}, rec@5={:.4f}'.format(
                        np.mean(cnt_acc), np.mean(cnt_rec1), np.mean(cnt_rec3), np.mean(cnt_rec5)))
                    break

    def forward(self, batch):
        matching_score = self.forward_neural(batch['context_text_ids'], batch['context_length'])
        kw_weight, predict_kw = tf.nn.top_k(matching_score, k=3)
        predict_kw = tf.reshape(predict_kw, [-1])
        predict_kw = tf.map_fn(lambda x: self.kw_list[x], predict_kw, dtype=tf.int64)
        predict_kw = tf.reshape(predict_kw, [-1, 3])
        embed_code = self.embedder(predict_kw)
        embed_code = tf.reduce_sum(embed_code, axis=1)
        embed_code = self.linear_transform(embed_code)

        source_embed = self.embedder(batch['source_text_ids'])
        target_embed = self.embedder(batch['target_text_ids'])
        target_embed = tf.reshape(target_embed, [-1, self.data_config._max_seq_len + 2, self.embedder.dim])
        target_length = tf.reshape(batch['target_length'], [-1])
        source_code = self.source_encoder(
            source_embed,
            sequence_length_minor=batch['source_length'],
            sequence_length_major=batch['source_utterance_cnt'])[1]  #
        target_code = self.target_encoder(
            target_embed,
            sequence_length=target_length)[1]
        target_kwcode = self.target_kwencoder(
            target_embed,
            sequence_length=target_length)[1]
        target_code = tf.concat([target_code[0], target_code[1], target_kwcode[0], target_kwcode[1]], -1)
        target_code = tf.reshape(target_code, [-1, 20, self.config._code_len])

        source_code = tf.concat([source_code, embed_code], -1)
        source_code = tf.expand_dims(source_code, 1)
        source_code = tf.tile(source_code, [1, 20, 1])
        feature_code = target_code * source_code
        feature_code = tf.reshape(feature_code, [-1, self.config._code_len])

        logits = self.linear_matcher(feature_code)
        logits = tf.reshape(logits, [-1, 20])
        labels = tf.one_hot(batch['label'], 20)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        ans = tf.arg_max(logits, -1)
        acc = tx.evals.accuracy(batch['label'], ans)
        rank = tf.nn.top_k(logits, k=20)[1]
        return loss, acc, rank

    def train(self):
        batch = self.iterator.get_next()
        kw_loss, kw_acc, _ = self.predict_keywords(batch)
        kw_saver = tf.train.Saver()
        loss, acc, _ = self.forward(batch)
        op_step = tf.Variable(0, name='retrieval_step')
        train_op = tx.core.get_train_op(loss, global_step=op_step, hparams=self.config.opt_hparams)
        max_val_acc = 0.
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            kw_saver.restore(sess, self.config._neural_save_path)
            saver = tf.train.Saver()
            for epoch_id in range(self.config._max_epoch):
                self.iterator.switch_to_train_data(sess)
                cur_step = 0
                cnt_acc = []
                while True:
                    try:
                        cur_step += 1
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                        loss, acc_ = sess.run([train_op, acc], feed_dict=feed)
                        cnt_acc.append(acc_)
                        if cur_step % 200 == 0:
                            print('batch {}, loss={}, acc1={}'.format(cur_step, loss, np.mean(cnt_acc[-200:])))
                    except tf.errors.OutOfRangeError:
                        break
                        
                self.iterator.switch_to_val_data(sess)
                cnt_acc, cnt_kwacc = [], []
                while True:
                    try:
                        feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                        acc_, kw_acc_ = sess.run([acc, kw_acc], feed_dict=feed)
                        cnt_acc.append(acc_)
                        cnt_kwacc.append(kw_acc_)
                    except tf.errors.OutOfRangeError:
                        mean_acc = np.mean(cnt_acc)
                        print('valid acc1={}, kw_acc1={}'.format(mean_acc, np.mean(cnt_kwacc)))
                        if mean_acc > max_val_acc:
                            max_val_acc = mean_acc
                            saver.save(sess, self.config._save_path)
                        break

    def test(self):
        batch = self.iterator.get_next()
        loss, acc, rank = self.forward(batch)
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(tf.tables_initializer())
            self.saver = tf.train.Saver()
            self.saver.restore(sess, self.config._save_path)
            self.iterator.switch_to_test_data(sess)
            rank_cnt = []
            while True:
                try:
                    feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
                    ranks, labels = sess.run([rank, batch['label']], feed_dict=feed)
                    for i in range(len(ranks)):
                        rank_cnt.append(np.where(ranks[i]==labels[i])[0][0])
                except tf.errors.OutOfRangeError:
                    rec = [0,0,0,0,0]
                    MRR = 0
                    for rank in rank_cnt:
                        for i in range(5):
                            rec[i] += (rank <= i)
                        MRR += 1 / (rank+1)
                    print('test rec1@20={:.4f}, rec3@20={:.4f}, rec5@20={:.4f}, MRR={:.4f}'.format(
                        rec[0]/len(rank_cnt), rec[2]/len(rank_cnt), rec[4]/len(rank_cnt), MRR/len(rank_cnt)))
                    break


    def retrieve_init(self, sess):
        data_batch = self.iterator.get_next()
        loss, acc, _ = self.forward(data_batch)
        self.corpus = self.data_config._corpus
        self.corpus_data = tx.data.MonoTextData(self.data_config.corpus_hparams)
        corpus_iterator = tx.data.DataIterator(self.corpus_data)
        batch = corpus_iterator.get_next()
        corpus_embed = self.embedder(batch['corpus_text_ids'])
        utter_code = self.target_encoder(corpus_embed, sequence_length=batch['corpus_length'])[1]
        utter_kwcode = self.target_kwencoder(corpus_embed, sequence_length=batch['corpus_length'])[1]
        utter_code = tf.concat([utter_code[0], utter_code[1], utter_kwcode[0], utter_kwcode[1]], -1)
        self.corpus_code = np.zeros([0, self.config._code_len])

        corpus_iterator.switch_to_dataset(sess)
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.config._save_path)
        feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
        while True:
            try:
                utter_code_ = sess.run(utter_code, feed_dict=feed)
                self.corpus_code = np.concatenate([self.corpus_code, utter_code_], axis=0)
            except tf.errors.OutOfRangeError:
                break
        self.keywords_embed = tf.nn.l2_normalize(self.embedder(self.kw_list), axis=1)
        self.kw_embedding = sess.run(self.keywords_embed)

        # predict keyword
        self.context_input = tf.placeholder(dtype=object, shape=(20))
        self.context_length_input = tf.placeholder(dtype=tf.int32, shape=(1))
        context_ids = tf.expand_dims(self.vocab.map_tokens_to_ids(self.context_input), 0)
        context_embed = self.embedder(context_ids)
        context_code = self.context_encoder(context_embed, sequence_length=self.context_length_input)[1]
        matching_score = self.predict_layer(context_code)
        self.candi_output =tf.nn.top_k(tf.squeeze(matching_score, 0), self.data_config._keywords_num)[1]

        # retrieve
        self.minor_length_input = tf.placeholder(dtype=tf.int32, shape=(1, 9))
        self.major_length_input = tf.placeholder(dtype=tf.int32, shape=(1))
        self.history_input = tf.placeholder(dtype=object, shape=(9, self.data_config._max_seq_len + 2))
        self.kw_input = tf.placeholder(dtype=tf.int32)
        history_ids = self.vocab.map_tokens_to_ids(self.history_input)
        history_embed = self.embedder(history_ids)
        history_code = self.source_encoder(tf.expand_dims(history_embed, axis=0),
                                           sequence_length_minor=self.minor_length_input,
                                           sequence_length_major=self.major_length_input)[1]
        self.next_kw_ids = self.kw_list[self.kw_input]
        embed_code = tf.expand_dims(self.embedder(self.next_kw_ids), 0)
        embed_code = self.linear_transform(embed_code)
        history_code = tf.concat([history_code, embed_code], 1)
        select_corpus = tf.cast(self.corpus_code, dtype=tf.float32)
        feature_code = self.linear_matcher(select_corpus * history_code)
        self.ans_output = tf.nn.top_k(tf.squeeze(feature_code,1), k=self.data_config._retrieval_candidates)[1]

    def retrieve(self, history_all, sess):
        history, seq_len, turns, context, context_len = history_all
        kw_candi = sess.run(self.candi_output, feed_dict={self.context_input: context,
                                                          self.context_length_input: [context_len]})
        for kw in kw_candi:
            tmp_score = sum(self.kw_embedding[kw] * self.kw_embedding[self.data_config._keywords_dict[self.target]])
            if tmp_score > self.score:
                self.score = tmp_score
                self.next_kw = self.data_config._keywords_candi[kw]
                break
        ans = sess.run(self.ans_output, feed_dict={self.history_input: history,
                                                   self.minor_length_input: [seq_len], self.major_length_input: [turns],
                                                   self.kw_input: self.data_config._keywords_dict[self.next_kw]})
        flag = 0
        reply = self.corpus[ans[0]]
        for i in ans:
            if i in self.reply_list:  # avoid repeat
                continue
            for wd in kw_tokenize(self.corpus[i]):
                if wd in self.data_config._keywords_candi:
                    tmp_score = sum(self.kw_embedding[self.data_config._keywords_dict[wd]] *
                                    self.kw_embedding[self.data_config._keywords_dict[self.target]])
                    if tmp_score > self.score:
                        reply = self.corpus[i]
                        self.score = tmp_score
                        self.next_kw = wd
                        flag = 1
                        break
            if flag == 0:
                continue
            break
        return reply
