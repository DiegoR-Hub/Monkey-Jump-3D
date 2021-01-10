# coding=utf-8
"""
Diego Ruiz, CC3501, 2020-2
Monkey Jump 3D on PyOpenGl
"""

import glfw
from OpenGL.GL import *
import numpy as np
import sys
import random

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as ls
import csv

"""ancho y alto de la ventana"""
width = 1200
height = 1200
def csvToPython():
    with open(sys.argv[1], newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def recibirCsv():
    """chequear que no hayan solo 1 en una fila ni que hayan solo ceros"""
    listaPosiciones = csvToPython()
    for fila in listaPosiciones:
        contador = 0
        for elemento in fila:
            if elemento == '1' or elemento == 'x':
                contador += 1
        if contador == 0:
            print("Error en el input, fila sin plataformas")
            return
        elif contador == 5:
            print("Error en el input, fila llena de plataformas, imposible saltar al siguiente nivel")
            return
    return listaPosiciones

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        self.camara = 1

    def controlObjPlataformas(self, ObjPlataformas):
        self.ObjPlataformas = ObjPlataformas
    def ortho(self):
        self.projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)
    def frustrum(self):
        self.projection = tr.frustum(-5, 5, -5, 5, 9, 100)
    def perspective(self):
        self.projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
    def controlSuzanne(self, suzanne):
        self.suzanne = suzanne

    def on_key(self, window, key, scancode, action, mods):
        if key == glfw.KEY_SPACE:
            self.fillPolygon = not controller.fillPolygon

        elif key == glfw.KEY_1:
            self.ortho()

        elif key == glfw.KEY_2:
            self.frustrum()

        elif key == glfw.KEY_3:
            self.perspective()

        elif key == glfw.KEY_A and action == glfw.PRESS:
            controller.suzanne.rotIzq()

        elif key == glfw.KEY_D and action == glfw.PRESS:
            controller.suzanne.rotDer()

        elif key == glfw.KEY_J and action == glfw.PRESS:
            controller.suzanne.jump(self.ObjPlataformas.Plataformas)

        elif key == glfw.KEY_T and action == glfw.PRESS:
            controller.suzanne.movimientoMasy()

        elif key == glfw.KEY_Y and action == glfw.PRESS:
            controller.suzanne.movimientoMenosy()

        elif key == glfw.KEY_U and action == glfw.PRESS:
            controller.suzanne.movimientoMasx()

        elif key == glfw.KEY_I and action == glfw.PRESS:
            controller.suzanne.movimientoMenosx()

        elif key == glfw.KEY_W and action == glfw.PRESS:
            controller.suzanne.moveForward(self.ObjPlataformas.Plataformas)

        elif key == glfw.KEY_ESCAPE:
            """solo con sys.exit no funciona, poner glfw.terminate()"""
            """necesario primero terminar la renderizacion y luego cerrar la ventana"""
            glfw.terminate()
            sys.exit()
def readFaceVertex(faceDescription):

    aux = faceDescription.split('/')

    assert len(aux[0]), "Vertex index has not been defined."

    faceVertex = [int(aux[0]), None, None]

    assert len(aux) == 3, "Only faces where its vertices require 3 indices are defined."

    if len(aux[1]) != 0:
        faceVertex[1] = int(aux[1])

    if len(aux[2]) != 0:
        faceVertex[2] = int(aux[2])

    return faceVertex
def readOBJ(filename, color):

    vertices = []
    normals = []
    textCoords= []
    faces = []

    with open(filename, 'r') as file:
        for line in file.readlines():
            aux = line.strip().split(' ')
            
            if aux[0] == 'v':
                vertices += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vn':
                normals += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vt':
                assert len(aux[1:]) == 2, "Texture coordinates with different than 2 dimensions are not supported"
                textCoords += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'f':
                N = len(aux)                
                faces += [[readFaceVertex(faceVertex) for faceVertex in aux[1:4]]]
                for i in range(3, N-1):
                    faces += [[readFaceVertex(faceVertex) for faceVertex in [aux[i], aux[i+1], aux[1]]]]

        vertexData = []
        indices = []
        index = 0

        # Per previous construction, each face is a triangle
        for face in faces:

            # Checking each of the triangle vertices
            for i in range(0,3):
                vertex = vertices[face[i][0]-1]
                normal = normals[face[i][2]-1]

                vertexData += [
                    vertex[0], vertex[1], vertex[2],
                    color[0], color[1], color[2],
                    normal[0], normal[1], normal[2]
                ]

            # Connecting the 3 vertices to create a triangle
            indices += [index, index + 1, index + 2]
            index += 3        

        return bs.Shape(vertexData, indices)

class Suzanne(object):
    def __init__(self):
        self.model = es.toGPUShape(shape=readOBJ('monkey.obj', (0.9, 0.6, 0.2)))
        self.posx = 0
        self.posy = 0
        self.posz = 0
        self.angulo = 0
        self.direction = 0
        self.dtheta = np.pi/4
        self.inicio = True
        self.sobre_plataforma = False
        self.velocidadz = 0
        self.velocidady = 0
        self.velocidadx = 0
        self.aceleracionx = -0.0009
        self.aceleraciony = -0.0009
        self.gravedad = -0.0009
        self.estadoVertical = "ground"
        self.transform = tr.uniformScale(1)
        self.transformInicial = tr.matmul([tr.uniformScale(0.5),tr.translate(0,0,2),tr.rotationX(np.pi/3), tr.rotationY(-np.pi/2)])
        self.perder = False
        self.ganar = False
        self.anguloGiro = 0
        self.totalCollideCannon = 0

    def jump(self, listaPlataformas):
        if self.estadoVertical == "ground":
            self.velocidadz = 0.085
            self.estadoVertical = "on air"
            self.posicionSalto()
        for plataforma in listaPlataformas:
            if (plataforma.sobre >= self.posz >= plataforma.bajo) and (
                    plataforma.bordedery >= self.posy >= plataforma.bordeizqy) \
                    and (plataforma.bordederx >= self.posx >= plataforma.bordeizqx) and plataforma.falsa:
                plataformas.Plataformas.remove(plataforma)

    def movimientoMasx(self):
        self.velocidadx = 0.073038
        self.aceleracionx = -0.0009

    def movimientoMasy(self):
        self.velocidady = 0.073038
        self.aceleraciony = -0.0009

    def movimientoMenosx(self):
        self.velocidadx = -0.073038
        self.aceleracionx = 0.0009

    def movimientoMenosy(self):
        self.velocidady = -0.073038
        self.aceleraciony = 0.0009

    def moveForward(self, listaPlataformas):
        if self.estadoVertical == "on air":
            return

        elif self.direction == 0:
            self.movimientoMenosy()
            self.jump(listaPlataformas)

        elif self.direction == 1:
            self.movimientoMasx()
            self.movimientoMenosy()
            self.jump(listaPlataformas)

        elif self.direction == 2:
            self.movimientoMasx()
            self.jump(listaPlataformas)

        elif self.direction == 3:
            self.movimientoMasx()
            self.movimientoMasy()
            self.jump(listaPlataformas)

        elif self.direction == 4:
            self.movimientoMasy()
            self.jump(listaPlataformas)
        elif self.direction == 5:
            self.movimientoMenosx()
            self.movimientoMasy()
            self.jump(listaPlataformas)

        elif self.direction == 6:
            self.movimientoMenosx()
            self.jump(listaPlataformas)

        elif self.direction == 7:
            self.movimientoMenosx()
            self.movimientoMenosy()
            self.jump(listaPlataformas)
        self.forwardDisponible = False

    def actualizarX(self):
        self.posx += self.velocidadx
        self.velocidadx += self.aceleracionx
        if self.velocidadx < 0 and self.aceleracionx < 0:
            self.velocidadx = 0
            self.forwardDisponible = True
        if self.velocidadx > 0 and self.aceleracionx > 0:
            self.velocidadx = 0
            self.forwardDisponible = True

    def actualizarY(self):
        self.posy += self.velocidady
        self.velocidady += self.aceleraciony
        if self.velocidady < 0 and self.aceleraciony < 0:
            self.velocidady = 0
        if self.velocidady > 0 and self.aceleraciony >0:
            self.velocidady = 0

    def actualizarZ(self):
        self.posz += self.velocidadz
        if self.estadoVertical == "on air":
            self.velocidadz += self.gravedad
        self.chequearSiChocaPiso()

    def posicionSalto(self):
        self.transformInicial = tr.matmul(
            [tr.uniformScale(0.5), tr.translate(0, 0, 2), tr.rotationX(np.pi / 6), tr.rotationY(-np.pi / 2)])

    def posicionTierra(self):
        self.transformInicial = tr.matmul(
            [tr.uniformScale(0.5), tr.translate(0, 0, 2), tr.rotationX(np.pi / 2), tr.rotationY(-np.pi / 2)])

    def chequearSiChocaPiso(self):
        if self.posz < 0:
            if self.inicio:
                self.posz = 0
                self.velocidadz = 0
                self.estadoVertical = "ground"
                self.posicionTierra()
            else:
                self.lose()

    def configurarViewProjectionModel(self, projection, view):
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, self.transform)
        self.actualizarTransformation()

    def giro(self, listaPlataformas):
        if self.anguloGiro == 2 * np.pi:
            self.jump(listaPlataformas)
            self.anguloGiro = 0
        else:
            self.anguloGiro += np.pi / 130
        self.actualizarTransformation()

    def draw(self, pipeline, projection, view, listaPlataformas, banana, listaCannons):
        if self.perder:
            self.giro(listaPlataformas)
        if self.ganar:
            self.jump(listaPlataformas)
        self.actualizarY()
        self.actualizarX()
        self.actualizarZ()
        self.actualizacionAngulo()
        self.configurarViewProjectionModel(projection, view)
        self.collidePlataformas(listaPlataformas)
        self.collideBanana(banana,listaPlataformas)
        self.collideCannon(listaCannons)
        pipeline.drawShape(suzanne.model)

    def drawWithoutChanges(self, pipeline, projection, view, listaPlataformas):
        self.configurarViewProjectionModel(projection, view)
        self.jump(listaPlataformas)
        pipeline.drawShape(suzanne.model)

    def actualizarTransformation(self):
        self.translate = tr.translate(self.posx, self.posy, self.posz)
        self.rotation = tr.rotationZ(self.angulo)
        self.transform = tr.matmul([self.translate, self.rotation, self.transformInicial, tr.rotationZ(-self.anguloGiro)])

    def actualizacionAngulo(self):
        self.angulo = self.direction * self.dtheta

    def rotIzq(self):
        if self.direction == 7:
            self.direction = 0
        else:
            self.direction += 1

    def rotDer(self):
        if self.direction == 0:
            self.direction = 7
        else:
            self.direction -= 1

    def win(self):
        self.ganar = True

    def lose(self):
        self.perder = True

    def collideCanhon(self, cannon):
        if cannon.posyder > self.posy > cannon.posyizq and cannon.poszarriba > self.posz > cannon.poszabajo:
            self.lose()

    def collidePlataformas(self, plataformas):
        if self.estadoVertical == "on air":
            for plataforma in plataformas:
                if self.velocidadz < 0:
                    if (plataforma.sobre >= self.posz >= plataforma.bajo) and (plataforma.bordedery >= self.posy >= plataforma.bordeizqy)\
                            and (plataforma.bordederx >= self.posx >= plataforma.bordeizqx):
                        self.posz = plataforma.sobre
                        self.posx = plataforma.posx
                        self.posy = plataforma.posy
                        self.estadoVertical = "ground"
                        self.inicio = False
                        self.velocidadz = 0

    def collideBanana(self, banana, listaPlataformas):
        if self.posx == banana.posx and self.posy == banana.posy and abs(self.posz - banana.posz) < 3.5:
            self.jump(listaPlataformas)
            self.win()

    def collideCannon(self, listaCannons):
        for cannon in listaCannons:
            self.collideSingleCannon(cannon)
        if self.totalCollideCannon >= 45:
            self.lose()

    def collideSingleCannon(self, cannon):
        if self.posx + 0.5 > cannon.posx > self.posx - 0.5:
            if self.posy + 0.5 > cannon.posy > self.posy - 0.5:
                if self.posz + 2.5 > cannon.posz > self.posz - 0.5:
                    self.totalCollideCannon += 1

class Suelo(object):
    def __init__(self):
        gpuSuelo = es.toGPUShape(bs.createColorCube(128/250,64/250,0))
        self.model = gpuSuelo
    def draw(self, pipeline, projection, view):
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([tr.translate(0, 0, -25), tr.uniformScale(50)]))
        pipeline.drawShape(self.model)
class Fondo(object):
    def __init__(self):
        gpuFondo = es.toGPUShape(bs.createTextureCube("bosque1.png"), GL_REPEAT, GL_NEAREST)
        self.model = gpuFondo
    def draw(self, pipeline, projection, view):
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([ tr.translate(0,0,25), tr.scale(50, 50, 60)]))
        pipeline.drawShape(self.model)
class Light(object):
    def __init__(self, height):
        self.height = height
        return
    def configurate(self):
        # Drawing shapes
        """setea los uniform del shader (ver codigo del shader en lightning_shaders.py"""
        """configuracion de la luz para su reflexion"""
        glUseProgram(pipeline.shaderProgram)
        """intensidad luz ambiental de la fuente"""
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        """intensidad luz difusa de la fuente"""
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        """intensidad luz especular de la fuente"""
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        """cantidad de luz ambiental que refleja el objeto, mas->mas cantidad de luz refleja, mas claro"""
        """ej 
        - glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0, 0, 0) no tiene luz ambiental, se ve negro si
        no fuera por difusa y especular
        - glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 1, 1, 1) tiene mucha luz ambiental, se ven muy claros
        los colores
        - mayor->colores base mas claros
        - menor->colores base mas oscuros
        - varian entre 0 y 1
         """
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
        """intensidad de brillo de luz especular
        - menor-> menos brillante
        - mayor-> mas brillante"""
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 0, 6, self.height)
        glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), 0, 0, 0)
        """
        alpha
        mas->mas brillo
        """
        glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.0004)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.04)
        glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.004)

class Plataforma(object):
    def __init__(self, posx=0, posy=0, posz=0, falsa = False):
        gpu_plataforma = es.toGPUShape(bs.createTextureCube("holi.png"), GL_REPEAT, GL_NEAREST)
        #gpu_plataforma = es.toGPUShape(bs.createColorCube(0.3,0.5,0.6))
        self.falsa = falsa
        self.model = gpu_plataforma
        self.translation = tr.translate(posx, posy, posz)
        self.posy = posy
        self.posx = posx
        self.posz = posz
        self.bajo = posz
        self.sobre = posz + 0.5  # real y mas mitad inferior
        self.bordederx = posx + 0.225
        self.bordeizqx = posx - 0.225
        self.bordedery = posy + 0.225
        self.bordeizqy = posy - 0.225

    def draw(self, pipeline, projection, view):
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([self.translation, tr.scale(2, 2, 0.5)]))
        pipeline.drawShape(self.model)

    def update(self, dt, mono):
        if mono.position == "on air":
            self.bajo -= dt * 0.9
            self.sobre -= dt * 0.9
            self.posy -= dt * 0.9
            self.model.transform = tr.translate(self.posx, self.posy, 0)
class PlataformaCreator(object):
    def __init__(self, listaPosiciones):
        self.listaPosiciones = listaPosiciones
        self.Plataformas = []
        self.banana = None
        self.Cannons = []
    def create(self):
        posz = 3
        posxbanana = 0
        posybanana = 0
        for fila in self.listaPosiciones:
            numerorand = random.randint(0,2)
            if numerorand == 1:
                self.Cannons.append(BehindCannon(posz))
            elif numerorand == 2:
                self.Cannons.append(FrontCannon(posz))
            for i in range(len(fila)):
                """contruir plataformas reales"""
                if fila[i] == '1':
                    if i == 0:
                        self.Plataformas.append(Plataforma(0, 6, posz, False))
                        posxbanana = 0
                        posybanana = 6
                    elif i == 1:
                        self.Plataformas.append(Plataforma(3, 3, posz, False))
                        posxbanana = 3
                        posybanana = 3
                    elif i == 2:
                        self.Plataformas.append(Plataforma(0, 0, posz, False))
                        posxbanana = 0
                        posybanana = 0
                    elif i == 3:
                        self.Plataformas.append((Plataforma(3, -3, posz, False)))
                        posxbanana = 3
                        posybanana = -3
                    elif i == 4:
                        self.Plataformas.append(Plataforma(0, -6, posz, False))
                        posxbanana = 0
                        posybanana = -6
                    """construir plataformas falsas"""
                elif fila[i] == 'x':
                    if i == 0:
                        self.Plataformas.append(Plataforma(0, 6, posz, True))
                        posxbanana = 0
                        posybanana = 6
                    elif i == 1:
                        self.Plataformas.append(Plataforma(3, 3, posz, True))
                        posxbanana = 3
                        posybanana = 3
                    elif i == 2:
                        self.Plataformas.append(Plataforma(0, 0, posz, True))
                        posxbanana = 0
                        posybanana = 0
                    elif i == 3:
                        self.Plataformas.append((Plataforma(3, -3, posz, True)))
                        posxbanana = 3
                        posybanana = -3
                    elif i == 4:
                        self.Plataformas.append(Plataforma(0, -6, posz, True))
                        posxbanana = 0
                        posybanana = -6
            posz += 3
        self.banana = Banana(posxbanana, posybanana, posz)
    def draw(self, pipelineCannon, pipelineplataforma, pipelineBanana, projection, view):
        self.banana.draw(pipelineBanana, projection, view)
        for plataforma in self.Plataformas:
            plataforma.draw(pipelineplataforma, projection, view)
        for cannon in self.Cannons:
            cannon.draw(pipelineCannon, projection, view)
class GameOver(object):
    def __init__(self, posx=0, posy=0, posz=0):
        self.model = es.toGPUShape(bs.createTextureCube("gameOver.png"), GL_REPEAT, GL_NEAREST)
        self.scaleConstant = 6
        self.scale = tr.uniformScale(self.scaleConstant)
        self.translate = tr.translate(posx, posy, posz)
        self.estado = "aumentando"
        self.cambioTamano = 0.01

    def cambiarTamano(self):
        if self.scaleConstant > 8:
            self.estado = "disminuyendo"
        elif self.scaleConstant < 6:
            self.estado = "aumentando"
        if self.estado == "aumentando":
            self.scaleConstant += self.cambioTamano
        else:
            self.scaleConstant -= self.cambioTamano

    def actualizarScale(self):
        self.scale = tr.uniformScale(self.scaleConstant)

    def draw(self, pipeline, projection, view):
        self.cambiarTamano()
        self.actualizarScale()
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([self.translate, self.scale]))
        pipeline.drawShape(self.model)
class WinnerScreen(object):
    def __init__(self, posx=0, posy=0, posz=0):
        self.model = es.toGPUShape(bs.createTextureCube("WinScreen.png"), GL_REPEAT, GL_NEAREST)
        self.scaleConstant = 6
        self.scale = tr.uniformScale(self.scaleConstant)
        self.translate = tr.translate(posx, posy, posz)
        self.estado = "aumentando"
        self.cambioTamano = 0.01

    def cambiarTamano(self):
        if self.scaleConstant > 8:
            self.estado = "disminuyendo"
        elif self.scaleConstant < 6:
            self.estado = "aumentando"
        if self.estado == "aumentando":
            self.scaleConstant += self.cambioTamano
        else:
            self.scaleConstant -= self.cambioTamano

    def actualizarScale(self):
        self.scale = tr.uniformScale(self.scaleConstant)

    def draw(self, pipeline, projection, view):
        self.cambiarTamano()
        self.actualizarScale()
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([self.translate, self.scale]))
        pipeline.drawShape(self.model)
class Banana(object):
    def __init__(self, posx=0, posy=0, posz=0):
        self.model = es.toGPUShape(bs.createTextureCube("banana.png"), GL_REPEAT, GL_NEAREST)
        self.translation = tr.translate(posx, posy, posz)
        self.posy = posy
        self.posx = posx
        self.posz = posz
        self.bajo = posz - 1.5
        self.sobre = posz + 1.5  # real y mas mitad inferior
        self.bordeder = posx + 1.5
        self.bordeizq = posx - 1.5

    def draw(self, pipeline, projection, view):
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([self.translation, tr.translate(0,0,2), tr.uniformScale(3) ]))
        pipeline.drawShape(self.model)
class BehindCannon(object):
    def __init__(self, posz, posx=3, posy=-8):
        self.model = es.toGPUShape(bs.createColorCube(1, 0, 0))
        self.posx = posx
        self.posy = posy
        self.posz = posz - 1.5
        self.translation = tr.translate(self.posx, self.posy, self.posz)
        self.velocidady = 0.053038

    def draw(self, pipeline, projection, view):
        self.actualizarY()
        self.actualizarTranslation()
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([self.translation]))
        pipeline.drawShape(self.model)

    def actualizarY(self):
        self.posy += self.velocidady
        if self.posy > 8:
            self.posy = -8

    def actualizarTranslation(self):
        self.translation = tr.translate(self.posx, self.posy, self.posz)
class FrontCannon(object):
    def __init__(self, posz, posx=0, posy=8):
        self.model = es.toGPUShape(bs.createColorCube(1, 0, 0))
        self.posx = posx
        self.posy = posy
        self.posz = posz - 1.5
        self.translation = tr.translate(self.posx, self.posy, self.posz)
        self.velocidady = -0.053038
    def draw(self, pipeline, projection, view):
        self.actualizarY()
        self.actualizarTranslation()
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE,
                           tr.matmul([self.translation]))
        pipeline.drawShape(self.model)
    def actualizarY(self):
        self.posy += self.velocidady
        if self.posy < -8:
            self.posy = 8
    def actualizarTranslation(self):
        self.translation = tr.translate(self.posx, self.posy, self.posz)

if __name__ == "__main__":

    controller = Controller()
    # Initialize glfw
    """si no inicia correctamente glfw se sale del codigo"""
    if not glfw.init():
        sys.exit()

    window = glfw.create_window(width, height, "Reading a *.obj file", None, None)

    if not window:
        glfw.terminate()
        sys.exit()
    """le pasamos la ventana a glfw"""
    """le decimos, estamos dibujando en este dominio"""
    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    """glfw solo genere cambios cuando tengo la ventana pinchada, puesta"""
    glfw.set_key_callback(window, controller.on_key)

    """pipeline texturas 3D"""
    pipelineTex3D = ls.SimpleTextureGouraudShaderProgram()
    # Defining shader programs
    #pipeline = ls.SimpleFlatShaderProgram()
    """ls es lightning shader, saca un pipeline"""
    """notar que SimpleGouraud es un objeto, tiene metodos"""
    pipeline = ls.SimplePhongShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()
    colorShaderProgram = es.SimpleModelViewProjectionShaderProgram()

    # Setting up the clear screen color
    """color de fondo"""
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    """define que cosas se pueden ver y que no"""
    glEnable(GL_DEPTH_TEST)
    """lista de posiciones para construir plataformas"""
    listaPosiciones = recibirCsv()
    """instancia modelos de la vista"""
    suzanne = Suzanne()

    fondo = Fondo()
    suelo = Suelo()
    plataformas = PlataformaCreator(listaPosiciones)
    plataformas.create()
    controller.controlSuzanne(suzanne)
    controller.controlObjPlataformas(plataformas)
    pruebaCannon = FrontCannon(3)
    gameOver = GameOver(0,0,6)
    winnerScreen = WinnerScreen(0,0,plataformas.banana.posz + 7)
    light = Light(plataformas.banana.posz)

    """obtiene el tiemp inicial de simulacion para usar dt's a futuro"""
    t0 = glfw.get_time()
    """definir un angulo de la camara"""
    camera_theta = -np.pi/2
    # Setting up the view transform
    """configuracion camara, coordenadas polares"""
    """radio camara"""
    """aumentar si quiero mas lejos y disminuir si quiero mas cerca"""
    R = 15
    """posx camara"""
    camX = R * np.sin(camera_theta)
    """posy camara"""
    camY = R * np.cos(camera_theta)
    """aumentar el tercer argumento hace que este mas arriba es Z"""
    viewPos = np.array([camX, camY, 4])
    """lookAt recibe donde esta la camara, hacia donde mira y la normal """
    view = tr.lookAt(viewPos, np.array([0, 0, 1]), np.array([0, 0, 1]))
    """posicion a ver"""
    at = np.array([0, 0, suzanne.posz + 5])

    while not glfw.window_should_close(window):
        at = np.array([0,0,suzanne.posz + 5])
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1
        if (glfw.get_key(window, glfw.KEY_B) == glfw.PRESS):
            controller.ortho()
            """mueve el angulo de la camara horario"""
            camera_theta = np.pi
            R = 20
            camX = R * np.sin(camera_theta)
            camY = R * np.cos(camera_theta)
            viewPos = np.array([camX, camY, 10])
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))

        elif (glfw.get_key(window, glfw.KEY_N) == glfw.PRESS):
            controller.perspective()
            """mueve el angulo de la camara horario"""
            camera_theta = 3*np.pi/4
            R = 30
            camX = R * np.sin(camera_theta)
            camY = R * np.cos(camera_theta)
            viewPos = np.array([camX, camY, 10])
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))

        elif (glfw.get_key(window, glfw.KEY_M) == glfw.PRESS):
            controller.ortho()
            R = 23
            camX = 0.1
            camY = 0.1
            """posicionado sobre el mono"""
            viewPos = np.array([camX, camY, suzanne.posz + 7])
            """mirando hacia el mono"""
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))

        elif (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            """mueve el angulo de la camara horario"""
            camera_theta -= 2 * dt
            R = 15
            camX = R * np.sin(camera_theta)
            camY = R * np.cos(camera_theta)
            viewPos = np.array([camX, camY, suzanne.posz + 7])
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))

        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            """mueve angulo de la camara antihorario"""
            camera_theta += 2 * dt
            R = 15
            camX = R * np.sin(camera_theta)
            camY = R * np.cos(camera_theta)
            viewPos = np.array([camX, camY, suzanne.posz + 7])
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))
        else:
            R = 15
            camX = R * np.sin(camera_theta)
            camY = R * np.cos(camera_theta)
            viewPos = np.array([camX, camY, suzanne.posz + 7])
            view = tr.lookAt(viewPos, at, np.array([0, 0, 1]))

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        """configuracion de brillo, reflejo, etc"""
        light.configurate()
        """dibujamos el estado actual de los modelos"""
        #fondo.draw(colorShaderProgram, controller.projection, view)
        #para texturas


        if suzanne.perder:
            suzanne.posx = 0
            suzanne.posy = 0
            suzanne.posz = -2
            suzanne.draw(pipeline, controller.projection, view, plataformas.Plataformas, plataformas.banana,
                         plataformas.Cannons)
            gameOver.draw(textureShaderProgram, controller.projection, view)
            fondo.draw(textureShaderProgram, controller.projection, view)
            glfw.swap_buffers(window)

        elif suzanne.ganar:
            suelo.draw(colorShaderProgram, controller.projection, view)
            suzanne.draw(pipeline, controller.projection, view, plataformas.Plataformas, plataformas.banana,
                         plataformas.Cannons)
            plataformas.draw(colorShaderProgram, textureShaderProgram, textureShaderProgram, controller.projection,
                             view)
            fondo.draw(textureShaderProgram, controller.projection, view)
            winnerScreen.draw(textureShaderProgram, controller.projection, view)
            glfw.swap_buffers(window)


        else:
            suelo.draw(colorShaderProgram, controller.projection, view)
            suzanne.draw(pipeline, controller.projection, view, plataformas.Plataformas, plataformas.banana,
                         plataformas.Cannons)
            plataformas.draw(colorShaderProgram, textureShaderProgram, textureShaderProgram,controller.projection, view)
            fondo.draw(textureShaderProgram, controller.projection, view)
            glfw.swap_buffers(window)

    glfw.terminate()
