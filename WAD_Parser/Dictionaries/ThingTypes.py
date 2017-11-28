# Dictionary for the Thing Type codes, from http://doom.wikia.com/wiki/Thing_types

things = {}
things['artifacts'] = [
    {'int':2023, 	'hex':'7E7', 	'version':'R', 	'radius':20, 	'sprite':'PSTR', 	'sequence':'A' 	   ,    'class':'AP' ,	'description':'Berserk'},
    {'int':2026, 	'hex':'7EA', 	'version':'S', 	'radius':20, 	'sprite':'PMAP', 	'sequence':'ABCDCB', 	'class':'AP', 	'description':'Computer map'},
    {'int':2014, 	'hex':'7DE', 	'version':'S', 	'radius':20, 	'sprite':'BON1', 	'sequence':'ABCDCB', 	'class':'AP' ,	'description':'Health potion'},
    {'int':2024, 	'hex':'7E8', 	'version':'S', 	'radius':20, 	'sprite':'PINS', 	'sequence':'ABCD' ,	    'class':'AP' ,	'description':'Invisibility'},
    {'int':2022, 	'hex':'7E6', 	'version':'R', 	'radius':20, 	'sprite':'PINV', 	'sequence':'ABCD' ,	    'class':'AP' ,	'description':'Invulnerability'},
    {'int':2045, 	'hex':'7FD', 	'version':'S', 	'radius':20, 	'sprite':'PVIS', 	'sequence':'AB' ,	    'class':'AP' ,	'description':'Light amplification visor'},
    {'int':83,      'hex':'53 ', 	'version':'2', 	'radius':20, 	'sprite':'MEGA', 	'sequence':'ABCD' ,	    'class':'AP' ,	'description':'Megasphere'},
    {'int':2013, 	'hex':'7DD', 	'version':'S', 	'radius':20, 	'sprite':'SOUL', 	'sequence':'ABCDCB', 	'class':'AP' ,	'description':'Soul sphere'},
    {'int':2015, 	'hex':'7DF', 	'version':'S', 	'radius':20, 	'sprite':'BON2', 	'sequence':'ABCDCB', 	'class':'AP' ,	'description':'Spiritual armor'}
    ]

things['powerups'] = [
    {'int':8,    	'hex':'8',    	'version':'S', 	'radius':20, 	'sprite':'BPAK', 	'sequence':'A', 	'class':'P', 	'description':'Backpack'},
    {'int':2019, 	'hex':'7E3', 	'version':'S', 	'radius':20, 	'sprite':'ARM2', 	'sequence':'AB',    'class':'P', 	'description':'Blue armor'},
    {'int':2018, 	'hex':'7E2', 	'version':'S', 	'radius':20, 	'sprite':'ARM1', 	'sequence':'AB',    'class':'P', 	'description':'Green armor'},
    {'int':2012, 	'hex':'7DC', 	'version':'S', 	'radius':20, 	'sprite':'MEDI', 	'sequence':'A', 	'class':'P', 	'description':'Medikit'},
    {'int':2025, 	'hex':'7E9', 	'version':'S', 	'radius':20, 	'sprite':'SUIT', 	'sequence':'A', 	'class':'P', 	'description':'Radiation suit'},
    {'int':2011, 	'hex':'7DB', 	'version':'S', 	'radius':20, 	'sprite':'STIM', 	'sequence':'A', 	'class':'P', 	'description':'Stimpack'}
]

things['weapons'] = [
    {'int':2006, 	'hex':'7D6', 	'version':'R', 	'radius':20, 	'sprite':'BFUG', 	'sequence':'A', 	'class':'WP', 	'description':'BFG 9000'},
    {'int':2002, 	'hex':'7D2', 	'version':'S', 	'radius':20, 	'sprite':'MGUN', 	'sequence':'A', 	'class':'WP', 	'description':'Chaingun'},
    {'int':2005, 	'hex':'7D5', 	'version':'S', 	'radius':20, 	'sprite':'CSAW', 	'sequence':'A', 	'class':'WP', 	'description':'Chainsaw'},
    {'int':2004, 	'hex':'7D4', 	'version':'R', 	'radius':20, 	'sprite':'PLAS', 	'sequence':'A', 	'class':'WP', 	'description':'Plasma rifle'},
    {'int':2003, 	'hex':'7D3', 	'version':'S', 	'radius':20, 	'sprite':'LAUN', 	'sequence':'A', 	'class':'WP', 	'description':'Rocket launcher'},
    {'int':2001, 	'hex':'7D1', 	'version':'S', 	'radius':20, 	'sprite':'SHOT', 	'sequence':'A', 	'class':'WP', 	'description':'Shotgun'},
    {'int':82,      'hex':'52',     'version':'2', 	'radius':20, 	'sprite':'SGN2', 	'sequence':'A', 	'class':'WP', 	'description':'Super shotgun'}
]

things['ammunitions'] = [
    {'int':2007,   	'hex':'7D7',  	'version':'S',    	'radius':20, 	'sprite':'CLIP', 	'sequence':'A',    	'class':'P',   	'description':'Ammo clip'},
    {'int':2048,   	'hex':'800',  	'version':'S',    	'radius':20, 	'sprite':'AMMO', 	'sequence':'A',    	'class':'P',   	'description':'Box of ammo'},
    {'int':2046,   	'hex':'7FE',  	'version':'S',    	'radius':20, 	'sprite':'BROK', 	'sequence':'A',    	'class':'P',   	'description':'Box of rockets'},
    {'int':2049,   	'hex':'801',  	'version':'S',    	'radius':20, 	'sprite':'SBOX', 	'sequence':'A',    	'class':'P',   	'description':'Box of shells'},
    {'int':2047,   	'hex':'7FF',  	'version':'R',    	'radius':20, 	'sprite':'CELL', 	'sequence':'A',    	'class':'P',   	'description':'Cell charge'},
    {'int':17, 	    'hex':'11',   	'version':'R',    	'radius':20, 	'sprite':'CELP', 	'sequence':'A',    	'class':'P',   	'description':'Cell charge pack'},
    {'int':2010,   	'hex':'7DA',  	'version':'S',    	'radius':20, 	'sprite':'ROCK', 	'sequence':'A',    	'class':'P',   	'description':'Rocket'},
    {'int':2008,   	'hex':'7D8',  	'version':'S',    	'radius':20, 	'sprite':'SHEL', 	'sequence':'A',    	'class':'P',   	'description':'Shotgun shells'}
]

things['keys'] = [
    {'int':5,  	'hex':'5',    	'version':'S',    	'radius':20, 	'sprite':'BKEY', 	'sequence':'AB',   	'class':'P',    	'description':'Blue keycard'},
    {'int':40, 	'hex':'28',   	'version':'R',    	'radius':20, 	'sprite':'BSKU', 	'sequence':'AB',   	'class':'P',    	'description':'Blue skull key'},
    {'int':13, 	'hex':'D',    	'version':'S',    	'radius':20, 	'sprite':'RKEY', 	'sequence':'AB',   	'class':'P',    	'description':'Red keycard'},
    {'int':38, 	'hex':'26',   	'version':'R',    	'radius':20, 	'sprite':'RSKU', 	'sequence':'AB',   	'class':'P',    	'description':'Red skull key'},
    {'int':6,  	'hex':'6',    	'version':'S',    	'radius':20, 	'sprite':'YKEY', 	'sequence':'AB',   	'class':'P',    	'description':'Yellow keycard'},
    {'int':39, 	'hex':'27',   	'version':'R',    	'radius':20, 	'sprite':'YSKU', 	'sequence':'AB',   	'class':'P',    	'description':'Yellow skull key'}
]

things['monsters'] = [
    {'int':68, 	    'hex':'44',   	    'version':'2',    	'radius':64, 	    'sprite':'BSPI', 	'sequence':'+',    	'class':'MO',   	    'description':'Arachnotron'},
    {'int':64, 	    'hex':'40',   	    'version':'2',    	'radius':20, 	    'sprite':'VILE', 	'sequence':'+',    	'class':'MO',   	    'description':'Arch-Vile'},
    {'int':3003,   	'hex':'BBB',  	    'version':'S',    	'radius':24, 	    'sprite':'BOSS', 	'sequence':'+',    	'class':'MO',   	    'description':'Baron of Hell'},
    {'int':3005,   	'hex':'BBD',  	    'version':'R',    	'radius':31, 	    'sprite':'HEAD', 	'sequence':'+',    	'class':'MO^',          'description':'Cacodemon'},
    {'int':65, 	    'hex':'41',   	    'version':'2',    	'radius':20, 	    'sprite':'CPOS', 	'sequence':'+',    	'class':'MO',   	    'description':'Chaingunner'},
    {'int':72, 	    'hex':'48',   	    'version':'2',    	'radius':16, 	    'sprite':'KEEN', 	'sequence':'A+',  	'class':'MO^',          'description':'Commander Keen'},
    {'int':16, 	    'hex':'10',   	    'version':'R',    	'radius':40, 	    'sprite':'CYBR', 	'sequence':'+',    	'class':'MO',   	    'description':'Cyberdemon'},
    {'int':3002,   	'hex':'BBA',  	    'version':'S',    	'radius':30, 	    'sprite':'SARG', 	'sequence':'+',    	'class':'MO',   	    'description':'Demon'},
    {'int':3004,   	'hex':'BBC',  	    'version':'S',    	'radius':20, 	    'sprite':'POSS', 	'sequence':'+',    	'class':'MO',   	    'description':'Former Human Trooper'},
    {'int':9,  	    'hex':'9',    	    'version':'S',    	'radius':20, 	    'sprite':'SPOS', 	'sequence':'+',    	'class':'MO',   	    'description':'Former Human Sergeant'},
    {'int':69, 	    'hex':'45',   	    'version':'2',    	'radius':24, 	    'sprite':'BOS2', 	'sequence':'+',    	'class':'MO',   	    'description':'Hell Knight'},
    {'int':3001,   	'hex':'BB9',  	    'version':'S',    	'radius':20, 	    'sprite':'TROO', 	'sequence':'+',    	'class':'MO',   	    'description':'Imp'},
    {'int':3006,   	'hex':'BBE',  	    'version':'R',    	'radius':16, 	    'sprite':'SKUL', 	'sequence':'+',    	'class':'MO^',          'description':'Lost Soul'},
    {'int':67, 	    'hex':'43',   	    'version':'2',    	'radius':48, 	    'sprite':'FATT', 	'sequence':'+',    	'class':'MO',   	    'description':'Mancubus'},
    {'int':71, 	    'hex':'47',   	    'version':'2',    	'radius':31, 	    'sprite':'PAIN', 	'sequence':'+',    	'class':'MO^',          'description':'Pain Elemental'},
    {'int':66, 	    'hex':'42',   	    'version':'2',    	'radius':20, 	    'sprite':'SKEL', 	'sequence':'+',    	'class':'MO',   	    'description':'Revenant'},
    {'int':58, 	    'hex':'3A',   	    'version':'S',    	'radius':30, 	    'sprite':'SARG', 	'sequence':'+',    	'class':'MO',   	    'description':'Spectre'},
    {'int':7,  	    'hex':'7',    	    'version':'R',    	'radius':128,       'sprite':'SPID',    'sequence':'+',    	'class':'MO',   	    'description':'Spider Mastermind'},
    {'int':84, 	    'hex':'54',   	    'version':'2',    	'radius':20, 	    'sprite':'SSWV', 	'sequence':'+',    	'class':'MO',   	    'description':'Wolfenstein SS'}
]

things['obstacles'] = [
    {'int':2035, 	'hex':'7F3',    'version':'S', 	'radius':10,  	'sprite':'BAR1', 	'sequence':'AB+', 	'class':'O',    	'description':'Barrel'},
    {'int':70, 	    'hex':'46',  	'version':'2', 	'radius':10,  	'sprite':'FCAN', 	'sequence':'ABC', 	'class':'O',    	'description':'Burning barrel'},
    {'int':43, 	    'hex':'2B',  	'version':'R', 	'radius':16,  	'sprite':'TRE1', 	'sequence':'A', 	'class':'O',    	'description':'Burnt tree'},
    {'int':35, 	    'hex':'23',  	'version':'S', 	'radius':16,  	'sprite':'CBRA', 	'sequence':'A', 	'class':'O',    	'description':'Candelabra'},
    {'int':41, 	    'hex':'29',  	'version':'R', 	'radius':16,  	'sprite':'CEYE', 	'sequence':'ABCB', 	'class':'O',    	'description':'Evil eye'},
    {'int':28, 	    'hex':'1C',  	'version':'R', 	'radius':16,  	'sprite':'POL2', 	'sequence':'A', 	'class':'O',    	'description':'Five skulls "shish kebab"'},
    {'int':42, 	    'hex':'2A',  	'version':'R', 	'radius':16,  	'sprite':'FSKU', 	'sequence':'ABC', 	'class':'O',    	'description':'Floating skull'},
    {'int':2028, 	'hex':'7EC',    'version':'S', 	'radius':16,  	'sprite':'COLU', 	'sequence':'A', 	'class':'O',    	'description':'Floor lamp'},
    {'int':53, 	    'hex':'35',  	'version':'R', 	'radius':16,  	'sprite':'GOR5', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging leg'},
    {'int':52, 	    'hex':'34',  	'version':'R', 	'radius':16,  	'sprite':'GOR4', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging pair of legs'},
    {'int':78, 	    'hex':'4E',  	'version':'2', 	'radius':16,  	'sprite':'HDB6', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging torso, brain removed'},
    {'int':75, 	    'hex':'4B',  	'version':'2', 	'radius':16,  	'sprite':'HDB3', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging torso, looking down'},
    {'int':77, 	    'hex':'4D',  	'version':'2', 	'radius':16,  	'sprite':'HDB5', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging torso, looking up'},
    {'int':76, 	    'hex':'4C',  	'version':'2', 	'radius':16,  	'sprite':'HDB4', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging torso, open skull'},
    {'int':50, 	    'hex':'32',  	'version':'R', 	'radius':16,  	'sprite':'GOR2', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging victim, arms out'},
    {'int':74, 	    'hex':'4A',  	'version':'2', 	'radius':16,  	'sprite':'HDB2', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging victim, guts and brain removed'},
    {'int':73, 	    'hex':'49',  	'version':'2', 	'radius':16,  	'sprite':'HDB1', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging victim, guts removed'},
    {'int':51, 	    'hex':'33',  	'version':'R', 	'radius':16,  	'sprite':'GOR3', 	'sequence':'A', 	'class':'O^',    	'description':'Hanging victim, one-legged'},
    {'int':49, 	    'hex':'31',  	'version':'R', 	'radius':16,  	'sprite':'GOR1', 	'sequence':'ABCB', 	'class':'O^',    	'description':'Hanging victim, twitching'},
    {'int':25, 	    'hex':'19',  	'version':'R', 	'radius':16,  	'sprite':'POL1', 	'sequence':'A', 	'class':'O',    	'description':'Impaled human'},
    {'int':54, 	    'hex':'36',  	'version':'R', 	'radius':32,  	'sprite':'TRE2', 	'sequence':'A', 	'class':'O',    	'description':'Large brown tree'},
    {'int':29, 	    'hex':'1D',  	'version':'R', 	'radius':16,  	'sprite':'POL3', 	'sequence':'AB', 	'class':'O',    	'description':'Pile of skulls and candles'},
    {'int':55, 	    'hex':'37',  	'version':'R', 	'radius':16,  	'sprite':'SMBT', 	'sequence':'ABCD', 	'class':'O',    	'description':'Short blue firestick'},
    {'int':56, 	    'hex':'38',  	'version':'R', 	'radius':16,  	'sprite':'SMGT', 	'sequence':'ABCD', 	'class':'O',    	'description':'Short green firestick'},
    {'int':31, 	    'hex':'1F',  	'version':'R', 	'radius':16,  	'sprite':'COL2', 	'sequence':'A', 	'class':'O',    	'description':'Short green pillar'},
    {'int':36, 	    'hex':'24',  	'version':'R', 	'radius':16,  	'sprite':'COL5', 	'sequence':'AB', 	'class':'O',    	'description':'Short green pillar with beating heart'},
    {'int':57, 	    'hex':'39',  	'version':'R', 	'radius':16,  	'sprite':'SMRT', 	'sequence':'ABCD', 	'class':'O',    	'description':'Short red firestick'},
    {'int':33, 	    'hex':'21',  	'version':'R', 	'radius':16,  	'sprite':'COL4', 	'sequence':'A', 	'class':'O',    	'description':'Short red pillar'},
    {'int':37, 	    'hex':'25',  	'version':'R', 	'radius':16,  	'sprite':'COL6', 	'sequence':'A', 	'class':'O',    	'description':'Short red pillar with skull'},
    {'int':86, 	    'hex':'56',  	'version':'2', 	'radius':16,  	'sprite':'TLP2', 	'sequence':'ABCD', 	'class':'O',    	'description':'Short techno floor lamp'},
    {'int':27, 	    'hex':'1B',  	'version':'R', 	'radius':16,  	'sprite':'POL4', 	'sequence':'A', 	'class':'O',    	'description':'Skull on a pole'},
    {'int':47, 	    'hex':'2F',  	'version':'R', 	'radius':16,  	'sprite':'SMIT', 	'sequence':'A', 	'class':'O',   	'description':'Stalagmite'},
    {'int':44, 	    'hex':'2C',  	'version':'R', 	'radius':16,  	'sprite':'TBLU', 	'sequence':'ABCD', 	'class':'O',    	'description':'Tall blue firestick'},
    {'int':45, 	    'hex':'2D',  	'version':'R', 	'radius':16,  	'sprite':'TGRN', 	'sequence':'ABCD', 	'class':'O',    	'description':'Tall green firestick'},
    {'int':30, 	    'hex':'1E',  	'version':'R', 	'radius':16,  	'sprite':'COL1', 	'sequence':'A', 	'class':'O',    	'description':'Tall green pillar'},
    {'int':46, 	    'hex':'2E',  	'version':'S', 	'radius':16,  	'sprite':'TRED', 	'sequence':'ABCD', 	'class':'O',    	'description':'Tall red firestick'},
    {'int':32, 	    'hex':'20',  	'version':'R', 	'radius':16,  	'sprite':'COL3', 	'sequence':'A', 	'class':'O',    	'description':'Tall red pillar'},
    {'int':85, 	    'hex':'55',  	'version':'2', 	'radius':16,  	'sprite':'TLMP', 	'sequence':'ABCD', 	'class':'O',    	'description':'Tall techno floor lamp'},
    {'int':48, 	    'hex':'30',  	'version':'S', 	'radius':16,  	'sprite':'ELEC', 	'sequence':'A', 	'class':'O',    	'description':'Tall techno pillar'},
    {'int':26, 	    'hex':'1A',  	'version':'R', 	'radius':16,  	'sprite':'POL6', 	'sequence':'AB', 	'class':'O',    	'description':'Twitching impaled human'},
]

things['decorations'] = [
    {'int':10, 	    'hex':'A',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'W', 	'class':'',   'description':'Bloody mess'},
    {'int':12, 	    'hex':'C',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'W', 	'class':'',   'description':'Bloody mess'},
    {'int':34, 	    'hex':'22',  	'version':'S', 	'radius':16,  	'sprite':'CAND', 	'sequence':'A', 	'class':'',   'description':'Candle'},
    {'int':22, 	    'hex':'16',  	'version':'R', 	'radius':31,  	'sprite':'HEAD', 	'sequence':'L', 	'class':'',   'description':'Dead cacodemon'},
    {'int':21, 	    'hex':'15',  	'version':'S', 	'radius':30,  	'sprite':'SARG', 	'sequence':'N', 	'class':'',   'description':'Dead demon'},
    {'int':18, 	    'hex':'12',  	'version':'S', 	'radius':20,  	'sprite':'POSS', 	'sequence':'L', 	'class':'',   'description':'Dead former human'},
    {'int':19, 	    'hex':'13',  	'version':'S', 	'radius':20,  	'sprite':'SPOS', 	'sequence':'L', 	'class':'',   'description':'Dead former sergeant'},
    {'int':20, 	    'hex':'14',  	'version':'S', 	'radius':20,  	'sprite':'TROO', 	'sequence':'M', 	'class':'',   'description':'Dead imp'},
    {'int':23, 	    'hex':'17',  	'version':'R', 	'radius':16,  	'sprite':'SKUL', 	'sequence':'K', 	'class':'',   'description':'Dead lost soul (invisible)'},
    {'int':15, 	    'hex':'F',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'N', 	'class':'',   'description':'Dead player'},
    {'int':62, 	    'hex':'3E',  	'version':'R', 	'radius':16,  	'sprite':'GOR5', 	'sequence':'A', 	'class':'^',  'description':'Hanging leg'},
    {'int':60, 	    'hex':'3C',  	'version':'R', 	'radius':16,  	'sprite':'GOR4', 	'sequence':'A', 	'class':'^',  'description':'Hanging pair of legs'},
    {'int':59, 	    'hex':'3B',  	'version':'R', 	'radius':16,  	'sprite':'GOR2', 	'sequence':'A', 	'class':'^',  'description':'Hanging victim, arms out'},
    {'int':61, 	    'hex':'3D',  	'version':'R', 	'radius':16,  	'sprite':'GOR3', 	'sequence':'A', 	'class':'^',  'description':'Hanging victim, one-legged'},
    {'int':63, 	    'hex':'3F',  	'version':'R', 	'radius':16,  	'sprite':'GOR1', 	'sequence':'ABCB', 	'class':'^',  'description':'Hanging victim, twitching'},
    {'int':79, 	    'hex':'4F',  	'version':'2', 	'radius':16,  	'sprite':'POB1', 	'sequence':'A', 	'class':'',   'description':'Pool of blood'},
    {'int':80, 	    'hex':'50',  	'version':'2', 	'radius':16,  	'sprite':'POB2', 	'sequence':'A', 	'class':'',   'description':'Pool of blood'},
    {'int':24, 	    'hex':'18',  	'version':'S', 	'radius':16,  	'sprite':'POL5', 	'sequence':'A', 	'class':'',   'description':'Pool of blood and flesh'},
    {'int':81, 	    'hex':'51',  	'version':'2', 	'radius':16,  	'sprite':'BRS1', 	'sequence':'A', 	'class':'',   'description':'Pool of brains'},
]

things['other'] = [
    {'int':88, 	    'hex':'58',  	'version':'2', 	'radius':16,  	'sprite':'BBRN', 	'sequence':'+', 	'class':'O',  'description':'Boss Brain'},
    {'int':11, 	    'hex':'B',   	'version':'S', 	'radius':20,  	'sprite':'none', 	'sequence':'-', 	'class':'',   'description':'Deathmatch start'},
    {'int':1, 	    'hex':'1',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'+', 	'class':'',   'description':'Player 1 start'},
    {'int':2, 	    'hex':'2',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'+', 	'class':'',   'description':'Player 2 start'},
    {'int':3, 	    'hex':'3',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'+', 	'class':'',   'description':'Player 3 start'},
    {'int':4, 	    'hex':'4',   	'version':'S', 	'radius':16,  	'sprite':'PLAY', 	'sequence':'+', 	'class':'',   'description':'Player 4 start'},
    {'int':89, 	    'hex':'59',  	'version':'2', 	'radius':20,  	'sprite':'none', 	'sequence':'-', 	'class':'',   'description':'Spawn shooter'},
    {'int':87, 	    'hex':'57',  	'version':'2', 	'radius':0,  	'sprite':'none', 	'sequence':'-', 	'class':'',   'description':'Spawn spot'},
    {'int':14, 	    'hex':'E',   	'version':'S', 	'radius':20,  	'sprite':'none', 	'sequence':'-', 	'class':'',   'description':'Teleport landing'},
]

all_things = []
for category in things.keys():
    all_things += things[category]

