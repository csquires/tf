
class Model(object):
    def load_data(self):
        raise NotImplementedError

    def add_placeholders(self):
        raise NotImplementedError

    def feed_dicts(self):
        raise NotImplementedError

    def add_model(self, input_data):
        raise NotImplementedError

    def add_loss_op(self, pred):
        raise NotImplementedError

    def run_epoch(self, sess, input_data, input_labels):
        raise NotImplementedError

    def fit(self, sess, input_data, input_labels):
        raise NotImplementedError

    def predict(self, sess, input_data, input_labels=None):
        raise NotImplementedError


class LanguageModel(Model):
    def add_embedding(self):
        raise NotImplementedError


class SkipGramModel:
    def __init__(self):
        pass

    def _create_placeholders(self):
        pass

    def _create_emedding(self):
        pass

    def _create_loss(self):
        pass

    def _create_optimizer(self):
        pass

