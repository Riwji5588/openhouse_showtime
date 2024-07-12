import pyglet
from pyglet.gl import *
from pyglet.extensions.gltf import gltf

# Load the .glb model using gltf
model_path = 'your_model.glb'
model = gltf.load(model_path)

# Create a window
window = pyglet.window.Window(width=1280, height=720)

# Set up OpenGL
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45.0, float(window.width) / window.height, 0.1, 100.0)
glMatrixMode(GL_MODELVIEW)

@window.event
def on_draw():
    # Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set up camera
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

    # Draw the model
    model.draw()

if __name__ == "__main__":
    pyglet.app.run()
