import numpy as np
import os
import argparse

#python generateMocapFile.py --amass_root= --datasets=CMU --datasets=HumanEva


def shuffleData(nrOfSequences):
    shuffleIdx = np.arange(0,nrOfSequences)
    np.random.shuffle(shuffleIdx)
    return shuffleIdx


parser = argparse.ArgumentParser(description='generate MoCap file from AMASS')
parser.add_argument('--amass_root',default='./amass', help='root directory of AMASS')
parser.add_argument('--datasets', '--list', action='append')
parser.add_argument('--saveName', type=str, default='mocapData')
parser.add_argument('--trainRatio', default=0.8)
parser.add_argument('--testRatio', default=0.1)

args = parser.parse_args()
datasets = args.datasets
root_dir = args.amass_root
saveName = args.saveName
trainRatio = args.trainRatio
testRatio = args.testRatio

mocapDict = {}
pose_counter = 0
for dataset in datasets:
    dataPath = os.path.join(root_dir, dataset)
    sequences = os.listdir(dataPath)

    for seqName in sequences:
        seqPath = os.path.join(dataPath, seqName)
        motions = os.listdir(seqPath)

        for motion in motions:
            seqMotion = np.load(os.path.join(seqPath, motion))

            for key in seqMotion.keys():
                if key in ['trans', 'poses', "mocap_framerate"]:
                    if key == 'poses':
                        keyName = 'pose'
                        pose_counter += 1
                    elif key == 'mocap_framerate':
                        keyName = 'framerate'
                    else:
                        keyName = key

                    motKey = '_'.join(motion.split('_')[:-1])
                    mocapKey = '_'.join([keyName, dataset.lower(), seqName, motKey])

                    mocapDict[mocapKey] = seqMotion[key]

if os.path.isfile('./datageneration/smpl_data/'+saveName+'.npz'):
    inK = None
    while not(inK == 'y') and not(inK == 'n'):
        print('warning, this file does already exist. If you want to overwrite it press y, else press n. Please note that the random train/test/val split will be overwritten!')
        inK = input()
        if inK == 'n':
            exit(-1)

np.savez('./datageneration/smpl_data/'+saveName+'.npz', **mocapDict)
shuffleIdx = shuffleData(pose_counter   )
np.save('./datageneration/resources/random_mocap_order_'+saveName+'.npy', shuffleIdx)
