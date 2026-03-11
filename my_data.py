# my data

import numpy as np

class DrivenSpinSystem:
    """
    Generates chaotic binary classification data using a driven frustrated spin system.
    """
    def __init__(self, n_features=10, n_spins=50, n_iterations=30, spectral_scaling=1.5, random_state=42):
        self.n_features = n_features
        self.n_spins = n_spins
        self.n_iterations = n_iterations
        self.spectral_scaling = spectral_scaling
        self.random_state = random_state
        
        # Initialize the random number generator
        self.rng = np.random.default_rng(self.random_state)
        
        # Build the chaotic core and projection matrices
        self._initialize_system()
        
    def _initialize_system(self):
        """Sets up the interaction matrix (W) and projection matrix (P)."""
        # 1. Define the Chaotic Core (The Interaction Matrix)
        # Asymmetric, fully connected weight matrix scaled by its spectral radius
        W_raw = self.rng.standard_normal((self.n_spins, self.n_spins))
        spectral_radius = np.max(np.abs(np.linalg.eigvals(W_raw)))
        
        # Scale > 1 pushes it into a chaotic regime (edge of chaos dynamics)
        self.W = W_raw / spectral_radius * self.spectral_scaling
        
        # 2. Projection Matrix
        # Maps the d-dimensional X into the n-dimensional spin space
        self.P = self.rng.standard_normal((self.n_features, self.n_spins))
        
    def generate(self, n_samples=1000, X=None):
        """
        Evolves the system and generates labels for the input data.
        If X is not provided, standard normal features are generated.
        """
        # 1. Handle Input Features (X)
        if X is None:
            X = self.rng.standard_normal((n_samples, self.n_features))
        else:
            # Ensure provided X matches the expected feature dimension
            if X.shape[1] != self.n_features:
                raise ValueError(f"Expected X to have {self.n_features} features, got {X.shape[1]}.")
            n_samples = X.shape[0]
            
        # 2. Calculate the external driving field applied by X at every time step
        External_Fields = X @ self.P 
        
        # 3. Evolve the System
        Spins = np.zeros((n_samples, self.n_spins))
        for _ in range(self.n_iterations):
            # Internal recurrent dynamics + External driving field X
            Spins = np.tanh(Spins @ self.W + External_Fields)
            
        # 4. Extract the Emergent Label
        magnetization = np.sum(Spins, axis=1)
        median_mag = np.median(magnetization)
        y = (magnetization > median_mag).astype(int)
        
        return X, y


class EasyData:
    """
    Generates a simple linearly separable dataset for binary classification.
    """
    def __init__(self, n_features=2, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
    def generate(self, n_samples=1000):
        X = self.rng.standard_normal((n_samples, self.n_features))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear decision boundary
        return X, y