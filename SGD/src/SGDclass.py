class SuperSGDClassifier:
    def __init__(self, lr=0.01, n_epochs=1000, threshold=0.5, fit_intercept=True):
        # threshold: umbral para convertir probabilidad en clase
        # fit_intercept, el sesgo
        self.lr = lr
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.hstack((intercept, X))
        return X

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        X = self._add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for epoch in range(self.n_epochs):
            for i in range(X.shape[0]):
                xi = X[i]
                yi = y[i]
                z = np.dot(xi, self.theta)
                h = self._sigmoid(z)
                gradient = (h - yi) * xi
                self.theta -= self.lr * gradient

            if epoch % 100 == 0:
                loss = self._loss(self._sigmoid(np.dot(X, self.theta)), y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        # Devuelve la probabilidad predicha (sigmoide de z)
        X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        # Devuelve 1 si probabilidad >= umbral, si no 0
        return (self.predict_proba(X) >= self.threshold).astype(int)
