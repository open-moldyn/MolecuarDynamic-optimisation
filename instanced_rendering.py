import moderngl
import gl_util

from window import Example
import window.pyqt5.window as window_cls


class InstancedRendering(Example):
    title = "Instanced Rendering"
    gl_version = (4, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader=gl_util.source("vertex.glsl",{}),
            fragment_shader='''
                #version 430

                in vec4 v_color;
                out vec4 f_color;

                void main() {
                    f_color = v_color;
                }
            ''',
        )

        self.ctx.point_size = 3.0

        self.scale = self.prog['Scale']
        self.n_a = self.prog['N_A']


    def render(self, a, b):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render(moderngl.POINTS,instances=self.npart)

def prepareRender():


    window = window_cls.Window(
        title=InstancedRendering.title,
        size=InstancedRendering.window_size,
        fullscreen=False,
        resizable=InstancedRendering.resizable,
        gl_version=InstancedRendering.gl_version,
        aspect_ratio=InstancedRendering.aspect_ratio,
        #aspect_ratio=InstancedRendering.window_size[0]/InstancedRendering.window_size[1],
        vsync=False,
        samples=1,
        cursor=True,
    )

    window.example = InstancedRendering(ctx=window.ctx, wnd=window)



    return window