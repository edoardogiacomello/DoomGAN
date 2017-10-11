'''
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import struct
from linedef import decode
import re


bit2object = {1:"Player start #1", 2:"Player start #2", 3:"Player start #3", 4:"Player start #4", 5:"BlueCard", 6:"YellowCard", 7:"SpiderMastermind", 8:"Backpack", 9:"ShotgunGuy", 10:"GibbedMarine", 11:"Deathmatch start", 12:"GibbedMarineExtra", 13:"RedCard", 14:"TeleportDest", 15:"DeadMarine", 16:"Cyberdemon", 17:"CellPack", 18:"DeadZombieMan", 19:"DeadShotgunGuy", 20:"DeadDoomImp", 21:"DeadDemon", 22:"DeadCacodemon", 23:"DeadLostSoul", 24:"Gibs", 25:"DeadStick", 26:"LiveStick", 27:"HeadOnAStick", 28:"HeadsOnAStick", 29:"HeadCandles", 30:"TallGreenColumn", 31:"ShortGreenColumn", 32:"TallRedColumn", 33:"ShortRedColumn", 34:"Candlestick", 35:"Candelabra", 36:"HeartColumn", 37:"SkullColumn", 38:"RedSkull", 39:"YellowSkull", 40:"BlueSkull", 41:"EvilEye", 42:"FloatingSkull", 43:"TorchTree", 44:"BlueTorch", 45:"GreenTorch", 46:"RedTorch", 47:"Stalagtite", 48:"TechPillar", 49:"BloodyTwitch", 50:"Meat2", 51:"Meat3", 52:"Meat4", 53:"Meat5", 54:"BigTree", 55:"ShortBlueTorch", 56:"ShortGreenTorch", 57:"ShortRedTorch", 58:"Spectre", 59:"NonsolidMeat2", 60:"NonsolidMeat4", 61:"NonsolidMeat3", 62:"NonsolidMeat5", 63:"NonsolidTwitch", 118:"ZBridge", 888:"MBFHelperDog", 1400:"Sound sequence thing (0)", 1401:"Sound sequence thing (1)", 1402:"Sound sequence thing (2)", 1403:"Sound sequence thing (3)", 1404:"Sound sequence thing (4)", 1405:"Sound sequence thing (5)", 1406:"Sound sequence thing (6)", 1407:"Sound sequence thing (7)", 1408:"Sound sequence thing (8)", 1409:"Sound sequence thing (9)", 1411:"Sound sequence thing (arg)", 1500:"Floor slope", 1501:"Ceiling slope", 1504:"Floor vertex height", 1505:"Ceiling vertex height", 2001:"Shotgun", 2002:"Chaingun", 2003:"RocketLauncher", 2004:"PlasmaRifle", 2005:"Chainsaw", 2006:"BFG9000", 2007:"Clip", 2008:"Shell", 2010:"RocketAmmo", 2011:"Stimpack", 2012:"Medikit", 2013:"Soulsphere", 2014:"HealthBonus", 2015:"ArmorBonus", 2018:"GreenArmor", 2019:"BlueArmor", 2022:"InvulnerabilitySphere", 2023:"Berserk", 2024:"BlurSphere", 2025:"RadSuit", 2026:"Allmap", 2028:"Column", 2035:"ExplosiveBarrel", 2045:"Infrared", 2046:"RocketBox", 2047:"Cell", 2048:"ClipBox", 2049:"ShellBox", 3001:"DoomImp", 3002:"Demon", 3003:"BaronOfHell", 3004:"ZombieMan", 3005:"Cacodemon", 3006:"LostSoul", 4001:"Player start #5", 4002:"Player start #6", 4003:"Player start #7", 4004:"Player start #8", 5001:"PointPusher", 5002:"PointPuller", 5004:"FS_Mapspot", 5006:"SkyCamCompat", 5010:"Pistol", 5050:"Stalagmite", 5061:"InvisibleBridge32", 5064:"InvisibleBridge16", 5065:"InvisibleBridge8", 9001:"MapSpot", 9013:"MapSpotGravity", 9024:"PatrolPoint", 9025:"SecurityCamera", 9026:"Spark", 9027:"RedParticleFountain", 9028:"GreenParticleFountain", 9029:"BlueParticleFountain", 9030:"YellowParticleFountain", 9031:"PurpleParticleFountain", 9032:"BlackParticleFountain", 9033:"WhiteParticleFountain", 9037:"BetaSkull", 9038:"ColorSetter", 9039:"FadeSetter", 9040:"MapMarker", 9041:"SectorFlagSetter", 9043:"TeleportDest3", 9044:"TeleportDest2", 9045:"WaterZone", 9046:"SecretTrigger", 9047:"PatrolSpecial", 9048:"SoundEnvironment", 9052:"StealthBaron", 9053:"StealthCacodemon", 9055:"StealthDemon", 9057:"StealthDoomImp", 9060:"StealthShotgunGuy", 9061:"StealthZombieMan", 9070:"InterpolationPoint", 9071:"PathFollower", 9072:"MovingCamera", 9073:"AimingCamera", 9074:"ActorMover", 9075:"InterpolationSpecial", 9076:"HateTarget", 9077:"UpperStackLookOnly", 9078:"LowerStackLookOnly", 9080:"SkyViewpoint", 9081:"SkyPicker", 9082:"SectorSilencer", 9083:"SkyCamCompat", 9100:"ScriptedMarine", 9101:"MarineFist", 9102:"MarineBerserk", 9103:"MarineChainsaw", 9104:"MarinePistol", 9105:"MarineShotgun", 9106:"MarineSSG", 9107:"MarineChaingun", 9108:"MarineRocket", 9109:"MarinePlasma", 9110:"MarineRailgun", 9111:"MarineBFG", 9200:"Decal", 9300:"PolyObject anchor", 9301:"PolyObject start spot (harmless)", 9302:"PolyObject start spot (crushing)", 9303:"PolyObject start spot (harmful)", 9500:"Floor line", 9501:"Ceiling line", 9502:"Floor tilt", 9503:"Ceiling tilt", 9510:"Copy floor slope", 9511:"Copy ceiling slope", 9982:"SecActEyesAboveC", 9983:"SecActEyesBelowC", 9988:"CustomSprite", 9989:"SecActHitFakeFloor", 9990:"InvisibleBridge", 9991:"CustomBridge", 9992:"SecActEyesSurface", 9993:"SecActEyesDive", 9994:"SecActUseWall", 9995:"SecActUse", 9996:"SecActHitCeil", 9997:"SecActExit", 9998:"SecActEnter", 9999:"SecActHitFloor", 14001:"AmbientSound", 14002:"AmbientSound", 14003:"AmbientSound", 14004:"AmbientSound", 14005:"AmbientSound", 14006:"AmbientSound", 14007:"AmbientSound", 14008:"AmbientSound", 14009:"AmbientSound", 14010:"AmbientSound", 14011:"AmbientSound", 14012:"AmbientSound", 14013:"AmbientSound", 14014:"AmbientSound", 14015:"AmbientSound", 14016:"AmbientSound", 14017:"AmbientSound", 14018:"AmbientSound", 14019:"AmbientSound", 14020:"AmbientSound", 14021:"AmbientSound", 14022:"AmbientSound", 14023:"AmbientSound", 14024:"AmbientSound", 14025:"AmbientSound", 14026:"AmbientSound", 14027:"AmbientSound", 14028:"AmbientSound", 14029:"AmbientSound", 14030:"AmbientSound", 14031:"AmbientSound", 14032:"AmbientSound", 14033:"AmbientSound", 14034:"AmbientSound", 14035:"AmbientSound", 14036:"AmbientSound", 14037:"AmbientSound", 14038:"AmbientSound", 14039:"AmbientSound", 14040:"AmbientSound", 14041:"AmbientSound", 14042:"AmbientSound", 14043:"AmbientSound", 14044:"AmbientSound", 14045:"AmbientSound", 14046:"AmbientSound", 14047:"AmbientSound", 14048:"AmbientSound", 14049:"AmbientSound", 14050:"AmbientSound", 14051:"AmbientSound", 14052:"AmbientSound", 14053:"AmbientSound", 14054:"AmbientSound", 14055:"AmbientSound", 14056:"AmbientSound", 14057:"AmbientSound", 14058:"AmbientSound", 14059:"AmbientSound", 14060:"AmbientSound", 14061:"AmbientSound", 14062:"AmbientSound", 14063:"AmbientSound", 14064:"AmbientSound", 14065:"AmbientSound", 14066:"SoundSequence", 14067:"AmbientSoundNoGravity", 14101:"MusicChanger", 14102:"MusicChanger", 14103:"MusicChanger", 14104:"MusicChanger", 14105:"MusicChanger", 14106:"MusicChanger", 14107:"MusicChanger", 14108:"MusicChanger", 14109:"MusicChanger", 14110:"MusicChanger", 14111:"MusicChanger", 14112:"MusicChanger", 14113:"MusicChanger", 14114:"MusicChanger", 14115:"MusicChanger", 14116:"MusicChanger", 14117:"MusicChanger", 14118:"MusicChanger", 14119:"MusicChanger", 14120:"MusicChanger", 14121:"MusicChanger", 14122:"MusicChanger", 14123:"MusicChanger", 14124:"MusicChanger", 14125:"MusicChanger", 14126:"MusicChanger", 14127:"MusicChanger", 14128:"MusicChanger", 14129:"MusicChanger", 14130:"MusicChanger", 14131:"MusicChanger", 14132:"MusicChanger", 14133:"MusicChanger", 14134:"MusicChanger", 14135:"MusicChanger", 14136:"MusicChanger", 14137:"MusicChanger", 14138:"MusicChanger", 14139:"MusicChanger", 14140:"MusicChanger", 14141:"MusicChanger", 14142:"MusicChanger", 14143:"MusicChanger", 14144:"MusicChanger", 14145:"MusicChanger", 14146:"MusicChanger", 14147:"MusicChanger", 14148:"MusicChanger", 14149:"MusicChanger", 14150:"MusicChanger", 14151:"MusicChanger", 14152:"MusicChanger", 14153:"MusicChanger", 14154:"MusicChanger", 14155:"MusicChanger", 14156:"MusicChanger", 14157:"MusicChanger", 14158:"MusicChanger", 14159:"MusicChanger", 14160:"MusicChanger", 14161:"MusicChanger", 14162:"MusicChanger", 14163:"MusicChanger", 14164:"MusicChanger", 14165:"MusicChanger", 32000:"DoomBuilderCamera",64:'Archvile',65:'ChaingunGuy',66:'Revenant',67:'Fatso',68:'Arachnotron',69:'HellKnight',70:'BurningBarrel',71:'PainElemental',72:'CommanderKeen',73:'HangNoGuts',74:'HangBNoBrain',75:'HangTLookingDown',76:'HangTSkull',77:'HangTLookingUp',78:'HangTNoBrain',79:'ColonGibs',80:'SmallBloodPool',81:'BrainStem',82:'SuperShotgun',83:'Megasphere',84:'WolfensteinSS',85:'TechLamp',86:'TechLamp2:',87:'BossTarget',88:'BossBrain',89:'BossEye',9050:'StealthArachnotron',9051:'StealthArchvile',9054:'StealthChaingunGuy',9056:'StealthHellKnight',9058:'StealthFatso',9059:'StealthRevenant',}


enemy=['SpiderMastermind','ShotgunGuy','Cyberdemon','DoomImp','Demon','BaronOfHell','ZombieMan','Cacodemon','LostSoul','StealthBaron','StealthCacodemon','StealthDemon','StealthDoomImp','StealthShotgunGuy','StealthZombieMan','MarineFist','MarineBerserk','MarineChainsaw','MarinePistol','MarineShotgun','MarineSSG','MarineChaingun','MarineRocket','MarinePlasma','MarineRailgun','MarineBFG','ScriptedMarine','Archvile','ChaingunGuy','Revenant','Fatso','Arachnotron','HellKnight','BossTarget','BossBrain','BossEye','StealthArachnotron','StealthArchvile','StealthChaingunGuy','StealthHellKnight','StealthFatso','StealthRevenant','PainElemental',]

weapon= ['Shotgun','Chaingun','RocketLauncher','PlasmaRifle','Chainsaw','BFG9000','Pistol','SuperShotgun']




ammo = ['Clip','Shell','RocketAmmo','RocketBox','Cell','ClipBox','ShellBox','CellPack']

health = ['Stimpack','Medikit','Soulsphere','HealthBonus','ArmorBonus','GreenArmor','BlueArmor','InvulnerabilitySphere','Berserk','BlurSphere','RadSuit','Allmap','Infrared','Megasphere']

environmental = ['ExplosiveBarrel']



class Wad(object):
    """Encapsulates the data found inside a WAD file"""

    def __init__(self, wadFile):
        """Each WAD files contains definitions for global attributes as well as map level attributes"""
        self.levels = []
        self.wad_format = 'DOOM' #Assume DOOM format unless 'BEHAVIOR' 
        with open(wadFile, "rb") as f:
            header_size = 12
            self.wad_type = f.read(4)[0]
            self.num_lumps = struct.unpack("<I", f.read(4))[0]
            data = f.read(struct.unpack("<I", f.read(4))[0] - header_size)

            current_level = Level(None) #The first few records of a WAD are not associated with a level

            lump = f.read(16) #Each offset is is part of a packet 16 bytes
            while len(lump) == 16:
                filepos = struct.unpack("<I", lump[0:4])[0] - header_size
                size = struct.unpack("<I", lump[4:8])[0]
                name = lump[8:16].decode('UTF-8').rstrip('\0')
                #print(name)
                if(re.match('E\dM\d|MAP\d\d', name)):
                    #Level nodes are named things like E1M1 or MAP01
                    if(current_level.is_valid()):
                        self.levels.append(current_level)
                    
                    current_level = Level(name)
                elif name == 'BEHAVIOR':
                    #This node only appears in Hexen formated WADs
                    self.wad_format = 'HEXEN'
                else:
                    current_level.lumps[name] = data[filepos:filepos+size]

                lump = f.read(16)
            if(current_level.is_valid()):
                self.levels.append(current_level)

        for level in self.levels:
            try:
                level.load(self.wad_format)
            except Exception:
                print "Failed to load level " + str(level)
class Level(object):
    """Represents a level inside a WAD which is a collection of lumps"""
    def __init__(self, name):
        self.name = name
        self.lumps = dict()
        self.vertices = []
        self.lower_left = None
        self.upper_right = None
        self.shift = None
        self.lines = []
        self.objects = []

    def is_valid(self):
        return self.name is not None and 'VERTEXES' in self.lumps and 'LINEDEFS' in self.lumps

    def normalize(self, point, padding=5):
        return (self.shift[0]+point[0]+padding,self.shift[1]+point[1]+padding)

    def load(self, wad_format):
        for vertex in packets_of_size(4, self.lumps['VERTEXES']):
            x,y = struct.unpack('<hh', vertex[0:4])
            self.vertices.append((x,y))

        self.lower_left = (min((v[0] for v in self.vertices)), min((v[1] for v in self.vertices)))
        self.upper_right = (max((v[0] for v in self.vertices)), max((v[1] for v in self.vertices)))

        self.shift = (0-self.lower_left[0],0-self.lower_left[1])
        self.midpt = ( (self.upper_right[0]+self.lower_left[0])*0.5+self.shift[0], (self.upper_right[1]+self.lower_left[1])*0.5+self.shift[1])
        
        print self.lower_left
        print self.upper_right
        print self.midpt
        
        packet_size = 16 if wad_format is 'HEXEN' else 14
        for data in packets_of_size(packet_size, self.lumps['LINEDEFS']):
            self.lines.append(Line(data))
        
        
        packet_size = 20 if wad_format is 'HEXEN' else 10
        for data in packets_of_size(packet_size, self.lumps['THINGS']):
            x,y,angle,type,spawn_flag = struct.unpack('<hhHHH',data)
            type = bit2object[type]
            self.objects.append( (x,y,angle,type,spawn_flag))
     
    def rotate(self,pt,midpt,angle):
        import math
        cosTheta = math.cos(angle*math.pi/180.0)
        sinTheta = math.sin(angle*math.pi/180.0)
        px = pt[0] - midpt[0]
        py = pt[1] - midpt[1]
        px_ = px*cosTheta - py*sinTheta
        py_ = px*sinTheta + py*cosTheta
        return (px_+midpt[0],py_+midpt[1])
    def save_svg(self):
        """ Scale the drawing to fit inside a 1024x1024 canvas (iPhones don't like really large SVGs even if they have the same detail) """
        import svgwrite
        view_box_size = self.normalize(self.upper_right, 10)
        if view_box_size[0] > view_box_size[1]:
            canvas_size = (1024, int(1024*(float(view_box_size[1])/view_box_size[0])))
        else:
            canvas_size = (int(1024*(float(view_box_size[0])/view_box_size[1])), 1024)
        
        dwg = svgwrite.Drawing(self.name+'.svg', profile='tiny', size=canvas_size , viewBox=('0 0 %d %d' % view_box_size))
        rot = 0
        for line in self.lines:
            a = self.normalize(self.vertices[line.a])
            b = self.normalize(self.vertices[line.b])
            a = self.rotate(a,self.midpt,rot)
            b = self.rotate(b,self.midpt,rot)
            
            if line.locked:
                dwg.add(dwg.line(a, b, stroke='#d00', stroke_width=20))
            elif line.teleport:
                dwg.add(dwg.line(a, b, stroke='#00d', stroke_width=20))
            elif line.door:
                dwg.add(dwg.line(a, b, stroke='#666', stroke_width=20))
            elif line.exit:                
                dwg.add(dwg.line(a, b, stroke='#0d0', stroke_width=20))
            elif line.is_one_sided():
                dwg.add(dwg.line(a, b, stroke='#333', stroke_width=10))
            else:
                dwg.add(dwg.line(a, b, stroke='#999', stroke_width=3))
        for obj in self.objects:
            x,y,angle,type,spawn = obj
            (x,y) = self.normalize((x,y))
            (x,y) = self.rotate((x,y),self.midpt,rot)
            if type in enemy:
                dwg.add(dwg.rect(insert=(x-16,y-24),size=(32,48), fill='#900',stroke='#900', stroke_width=5,transform='rotate({},{},{})'.format(angle+rot,x,y)))
            elif type in weapon:
                dwg.add(dwg.circle(center=(x,y),r=16,  fill='#900',stroke='#900', stroke_width=5))
            
            elif type in ammo:
                dwg.add(dwg.circle(center=(x,y),r=16,  fill='#090',stroke='#090', stroke_width=5))
            
            elif type in health:
                dwg.add(dwg.circle(center=(x,y),r=16, fill='#009',stroke='#009', stroke_width=5))
            
            elif type in environmental:
                dwg.add(dwg.ellipse(center=(x,y),r=(12,24),  fill='#900',stroke='#900', stroke_width=5) )               
            
            elif 'Card' in type or ('Skull' in type and 'Floating' not in type):
                dwg.add(dwg.ellipse(center=(x,y),r=(12,24), fill='#090',stroke='#090', stroke_width=5)  )          
            
            elif 'Player start' in type:
                dwg.add(dwg.ellipse(center=(x,y),r=(12,24), fill='#009',stroke='#009', stroke_width=5))
            elif 'TeleportDest' in type:
                dwg.add(dwg.rect(insert=(x-12,y-12),size=(24,24), fill='#009',stroke='#009', stroke_width=5,transform='rotate({},{},{})'.format(angle,x,y)))
            else :
                dwg.add(dwg.rect(insert=(x-12,y-12),size=(24,24), fill='#090',stroke='#090', stroke_width=5,transform='rotate({},{},{})'.format(angle,x,y)))
            
        dwg.save()

class Line(object):
    """Represents a Linedef inside a WAD"""
    """Represents a Linedef inside a WAD"""
    def __init__(self,data):
        self.a, self.b, self.line_flags, self.line_type, self.sector,self.left_side, self.right_side  = struct.unpack('<hhhhhhh', data[0:14])
        self.line_type = decode(self.line_type)
        self.door =  'DOOR' in self.line_type
        self.exit =  'EXIT' in self.line_type
        self.teleport =  'TELEPORT' in self.line_type
        self.locked = 'BLU' in self.line_type or 'RED' in self.line_type or 'YEL' in self.line_type
        

    def is_one_sided(self):
        return self.left_side == -1 or self.right_side == -1

def packets_of_size(n, data):
    size = len(data)
    index = 0
    while index < size:
        yield data[index : index+n]
        index = index + n
    return

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        wad = Wad(sys.argv[1])
        for level in wad.levels:
            level.save_svg()
    else:
        print('You need to pass a WAD file as the only argument')