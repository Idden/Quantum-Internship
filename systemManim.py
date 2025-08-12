import numpy as np
import qutip as qt
from manim import *

class System:

    def __init__(self, system_data):


        self.sx = qt.sigmax()
        self.sy = qt.sigmay()
        self.sz = qt.sigmaz()
        self.I = qt.qeye(2)

        self.system_data = system_data
        

        if system_data['system'] == 'single_qubit':
            self.eps = system_data['eps']
            self.delta = system_data['delta']
            self.theta = system_data['theta']
            self.phi = system_data['phi']
            self.psi0 = (np.cos(self.theta/2) * qt.basis(2, 0) + np.exp(1j * self.phi) * np.sin(self.theta/2) * qt.basis(2, 1))

        elif system_data['system'] == 'multi_qubit':
            self.Js = system_data['Js']
            self.Hs = system_data['Hs']
            self.N = len(self.Hs)
            psi0 = qt.rand_ket(2**self.N)
            psi0.dims = [[2]*self.N, 1]
            self.psi0 = psi0
        

    def hamiltonian(self):
        if self.system_data['system'] == 'single_qubit':
            H = self.eps * self.sz / 2 + self.delta * self.sx / 2
        if self.system_data['system'] == 'multi_qubit':
            H = 0
            for j, hs in enumerate(self.Hs):
                
                H += qt.tensor([self.I]*j + [hs*self.sz] + [self.I]*(self.N-j-1))
                
            for j, js in enumerate(self.Js):

                H += js*qt.tensor([self.I]*j + [self.sx, self.sx] + [self.I]*(self.N-j-2))

        return H


    def evolve(self):
        tlist = np.linspace(0, 3, 20)
        H = self.hamiltonian()

        return qt.sesolve(H, self.psi0, tlist)
    
    def expectation_values(self):
        evol = self.evolve()
        if self.system_data['system'] == 'single_qubit':
            return {
                'sx': qt.expect(self.sx, evol.states),
                'sy': qt.expect(self.sy, evol.states),
                'sz': qt.expect(self.sz, evol.states)
            }
        if self.system_data['system'] == 'multi_qubit':
            return {
                'sx': [qt.expect(qt.tensor([self.sx] + [self.I]*(self.N-1)), state) for state in evol.states],
                'sy': [qt.expect(qt.tensor([self.sy] + [self.I]*(self.N-1)), state) for state in evol.states],
                'sz': [qt.expect(qt.tensor([self.sz] + [self.I]*(self.N-1)), state) for state in evol.states]
            }
        


class BlochSphere(ThreeDScene, System):

    def __init__(self, **kwargs):
        system_data = {
            'system': 'single_qubit',
            'eps': 1.0,
            'delta': 1.0,
            'theta': PI / 3,
            'phi': PI / 4
        }
        
        System.__init__(self, system_data)
        
        ThreeDScene.__init__(self, **kwargs)

    def construct(self):

        # camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # vector coordinates
        expect = self.expectation_values()
        sx = expect['sx']
        sy = expect['sy']
        sz = expect['sz']

        # mobjects
        axes = ThreeDAxes(x_range=[-1.5, 1.5, 1], x_length=6,
                          y_range=[-1.5, 1.5, 1], y_length=6,
                          z_range=[-1.5, 1.5, 1], z_length=6, 
                          tips=False)
        
        sphere = Sphere(radius=2, resolution=(20, 20), fill_opacity=0.05, stroke_color="BLUE_C")

        vector = Arrow3D(start=[0, 0, 0],
                         end=np.array([2*sx[0], 2*sy[0], 2*sz[0]]),
                         color=RED,
                         stroke_width=6)

        # axes kets
        x_label_pos = MathTex("|+\\rangle").scale(1).move_to(axes.x_axis.get_end() + 0.7*RIGHT).set_color("YELLOW")
        x_label_neg = MathTex("|-\\rangle").scale(1).move_to(axes.x_axis.get_start() - 0.7*RIGHT).set_color("YELLOW")

        y_label_pos = MathTex("|i\\rangle").scale(1).move_to(axes.y_axis.get_end() + 0.5*UP).set_color("YELLOW")
        y_label_neg = MathTex("|-i\\rangle").scale(1).move_to(axes.y_axis.get_start() - 0.9*UP).set_color("YELLOW")

        z_label_pos = MathTex("|0\\rangle").scale(1).move_to(axes.z_axis.get_end() + 0.4*OUT).set_color("YELLOW")
        z_label_neg = MathTex("|1\\rangle").scale(1).move_to(axes.z_axis.get_start() - 0.4*OUT).set_color("YELLOW")

        # animation code
        self.play(Write(axes), Write(sphere))

        self.add_fixed_orientation_mobjects(x_label_pos, x_label_neg,
                                            y_label_pos, y_label_neg,
                                            z_label_pos, z_label_neg)
        
        self.play(Write(x_label_pos), Write(x_label_neg), 
                  Write(y_label_pos), Write(y_label_neg), 
                  Write(z_label_pos), Write(z_label_neg))
        
        for i in range(1, len(sx)):
            self.play(
                vector.animate.put_start_and_end_on([0, 0, 0], np.array([2*sx[i], 2*sy[i], 2*sz[i]])),
                run_time=0.1
            )

        self.wait()
