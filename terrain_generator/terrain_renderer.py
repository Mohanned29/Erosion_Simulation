import pyglet
from pyglet import gl
import numpy as np
import ctypes

# -------------------------------
# Shader source code (GLSL 330 core)
# -------------------------------
vertex_shader_source = b"""
#version 330 core
in vec3 position;
in vec3 color;
uniform mat4 mvp;
out vec3 fragColor;
void main()
{
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = color;
}
"""

fragment_shader_source = b"""
#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main()
{
    outColor = vec4(fragColor, 1.0);
}
"""

# -------------------------------
# Shader compilation helpers
# -------------------------------
def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    # Create a ctypes pointer to our source
    src_buffer = ctypes.create_string_buffer(source)
    src_ptr = ctypes.cast(ctypes.pointer(ctypes.pointer(src_buffer)),
                          ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    length = ctypes.c_int(len(source))
    gl.glShaderSource(shader, 1, src_ptr, ctypes.byref(length))
    gl.glCompileShader(shader)

    # Check compile status
    status = ctypes.c_int()
    gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int()
        gl.glGetShaderiv(shader, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        gl.glGetShaderInfoLog(shader, log_length, None, log)
        raise RuntimeError("Shader compilation failed:\n" + log.value.decode())
    return shader

def compile_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    # Check linking status
    status = ctypes.c_int()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int()
        gl.glGetProgramiv(program, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        gl.glGetProgramInfoLog(program, log_length, None, log)
        raise RuntimeError("Program linking failed:\n" + log.value.decode())

    # Cleanup shaders after linking
    gl.glDetachShader(program, vertex_shader)
    gl.glDetachShader(program, fragment_shader)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    return program

# -------------------------------
# Matrix utility functions
# -------------------------------
def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m

def rotation_x(angle):
    a = np.radians(angle)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = np.cos(a)
    m[1, 2] = -np.sin(a)
    m[2, 1] = np.sin(a)
    m[2, 2] = np.cos(a)
    return m

# -------------------------------
# TerrainRenderer class using modern OpenGL
# -------------------------------
class TerrainRenderer:
    def __init__(self, heightmap):
        self.heightmap = heightmap

        # Create a window with an OpenGL 3.3 core context
        config = gl.Config(major_version=3, minor_version=3, depth_size=24,
                           double_buffer=True, forward_compatible=True)
        self.window = pyglet.window.Window(width=800, height=600,
                                            caption="Terrain Renderer", config=config)
        self.window.on_draw = self.on_draw

        # Compile shader program and get uniform location
        self.shader_program = compile_program(vertex_shader_source, fragment_shader_source)
        self.mvp_location = gl.glGetUniformLocation(self.shader_program, b"mvp")

        # Prepare vertex data (positions and colors)
        self.vertex_count = 0
        self.vao = gl.GLuint(0)
        self.vbo = gl.GLuint(0)
        self.build_terrain()

        # Enable depth testing and set clear color
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.5, 0.7, 1.0, 1.0)

    def build_terrain(self):
        rows, cols = self.heightmap.shape
        vertices = []  # position data
        colors = []    # color data

        # Create two triangles per grid cell
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Calculate heights (amplified for visibility)
                z1 = self.heightmap[i, j] * 50
                z2 = self.heightmap[i+1, j] * 50
                z3 = self.heightmap[i, j+1] * 50
                z4 = self.heightmap[i+1, j+1] * 50

                # Colors based on height (green channel)
                c1 = (0.0, z1/50, 0.0)
                c2 = (0.0, z2/50, 0.0)
                c3 = (0.0, z3/50, 0.0)
                c4 = (0.0, z4/50, 0.0)

                # First triangle: vertices at (i,j), (i+1,j), (i,j+1)
                vertices.extend([i,    j,    z1,
                                 i+1,  j,    z2,
                                 i,    j+1,  z3])
                colors.extend(list(c1) + list(c2) + list(c3))

                # Second triangle: vertices at (i+1,j), (i+1,j+1), (i,j+1)
                vertices.extend([i+1,  j,    z2,
                                 i+1,  j+1,  z4,
                                 i,    j+1,  z3])
                colors.extend(list(c2) + list(c4) + list(c3))

        self.vertex_count = int(len(vertices) // 3)

        # Convert lists to numpy arrays
        vertex_data = np.array(vertices, dtype=np.float32)
        color_data = np.array(colors, dtype=np.float32)
        # Interleave position and color: [pos(x,y,z), color(r,g,b)] per vertex
        interleaved = np.empty(self.vertex_count * 6, dtype=np.float32)
        interleaved[0::6] = vertex_data[0::3]
        interleaved[1::6] = vertex_data[1::3]
        interleaved[2::6] = vertex_data[2::3]
        interleaved[3::6] = color_data[0::3]
        interleaved[4::6] = color_data[1::3]
        interleaved[5::6] = color_data[2::3]

        # Generate and bind VAO and VBO
        vao = gl.GLuint(0)
        gl.glGenVertexArrays(1, ctypes.byref(vao))
        self.vao = vao
        vbo = gl.GLuint(0)
        gl.glGenBuffers(1, ctypes.byref(vbo))
        self.vbo = vbo

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        size = interleaved.nbytes
        array_type = (gl.GLfloat * len(interleaved))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, array_type(*interleaved), gl.GL_STATIC_DRAW)

        # Specify layout of the vertex data
        # Attribute 0: position (3 floats)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 6 * ctypes.sizeof(gl.GLfloat), ctypes.c_void_p(0))
        # Attribute 1: color (3 floats)
        offset = 3 * ctypes.sizeof(gl.GLfloat)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 6 * ctypes.sizeof(gl.GLfloat), ctypes.c_void_p(offset))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def on_draw(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.shader_program)

        # Build a Model-View-Projection matrix
        aspect = self.window.width / self.window.height
        proj = perspective(45, aspect, 0.1, 1000.0)
        # Simulate translation and rotation (similar to your legacy calls):
        # First translate, then rotate about the X-axis by 45 degrees.
        view = np.dot(translate(-50, -50, -150), rotation_x(45))
        mvp = np.dot(proj, view)

        # Upload the MVP matrix to the shader
        mvp_flat = mvp.flatten()
        mvp_ptr = mvp_flat.ctypes.data_as(ctypes.POINTER(gl.GLfloat))
        gl.glUniformMatrix4fv(self.mvp_location, 1, gl.GL_FALSE, mvp_ptr)

        # Draw the terrain using our VAO
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def run(self):
        pyglet.app.run()
