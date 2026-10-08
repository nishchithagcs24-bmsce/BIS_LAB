
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(10, activation="tanh"),
        tf.keras.layers.Dense(10, activation="tanh"),
        tf.keras.layers.Dense(10, activation="tanh"),
        tf.keras.layers.Dense(1)
    ])

N = 100
x_r = tf.reshape(tf.linspace(0.0, 1.0, N), (-1, 1))
x_b = tf.constant([[0.0], [1.0]], dtype=tf.float32)

def physics_loss(model, x):
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u = model(x)
        u_x = tape1.gradient(u, x)
    u_xx = tape2.gradient(u_x, x)
    g = -(2 * np.pi)**2 * tf.sin(2 * np.pi * x)
    return tf.reduce_mean(tf.square(u_xx - g))

def boundary_loss(model):
    return tf.reduce_mean(tf.square(model(x_b)))

def total_loss(model):
    return physics_loss(model, x_r) + boundary_loss(model)

def get_weights(model):
    return np.concatenate([w.numpy().flatten() for w in model.trainable_variables])

def set_weights(model, vector):
    index = 0
    for variable in model.trainable_variables:
        size = np.prod(variable.shape)
        variable.assign(vector[index:index + size].reshape(variable.shape))
        index += size

def get_gradient(model):
    with tf.GradientTape() as tape:
        loss = total_loss(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_vector = np.concatenate([g.numpy().flatten() for g in gradients])
    return gradient_vector, loss.numpy()

NUM_PARTICLES = 20
MAX_ITER = 500
beta = 0.9
c1 = 0.08
c2 = 0.5
alpha = 0.005

particles = []
velocities = []
pbests = []
pbest_losses = []

base_model = create_model()
dimension = len(get_weights(base_model))

for i in range(NUM_PARTICLES):
    model = create_model()
    position = get_weights(model)
    velocity = np.zeros(dimension)
    loss = total_loss(model).numpy()

    particles.append(position)
    velocities.append(velocity)
    pbests.append(position.copy())
    pbest_losses.append(loss)

best_index = np.argmin(pbest_losses)
gbest = pbests[best_index].copy()
gbest_loss = pbest_losses[best_index]

loss_history = []

for iteration in range(MAX_ITER):
    decay = 1.0 - iteration / MAX_ITER
    current_c1 = c1 * decay
    current_c2 = c2 * decay

    for i in range(NUM_PARTICLES):
        model = create_model()
        set_weights(model, particles[i])

        gradient, current_loss = get_gradient(model)

        r1 = np.random.rand(dimension)
        r2 = np.random.rand(dimension)

        velocities[i] = (
            beta * velocities[i]
            + current_c1 * r1 * (pbests[i] - particles[i])
            + current_c2 * r2 * (gbest - particles[i])
            - alpha * gradient
        )

        particles[i] += velocities[i]

        set_weights(model, particles[i])
        new_loss = total_loss(model).numpy()

        if new_loss < pbest_losses[i]:
            pbests[i] = particles[i].copy()
            pbest_losses[i] = new_loss

        if new_loss < gbest_loss:
            gbest = particles[i].copy()
            gbest_loss = new_loss

    loss_history.append(gbest_loss)

    if iteration % 50 == 0:
        print(f"Iteration {iteration}: Loss = {gbest_loss:.8f}")

final_model = create_model()
set_weights(final_model, gbest)

x_test = np.linspace(0, 1, 200).reshape(-1, 1)
prediction = final_model(
    tf.convert_to_tensor(x_test, dtype=tf.float32)
).numpy()

exact = np.sin(2 * np.pi * x_test)

plt.figure(figsize=(8, 5))
plt.plot(x_test, exact, label="Exact")
plt.plot(x_test, prediction, "--", label="PSO-BP-CD PINN")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.semilogy(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()
```
