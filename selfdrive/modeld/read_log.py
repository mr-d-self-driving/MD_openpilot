from tools.lib.logreader import LogReader
for ev in LogReader("/home/adas/Downloads/d34c14daa88a1e86_000000ca--7c5d326170--7--rlog.bz2"):
    if ev.which() == "liveLocationKalmanDEPRECATED":
        # if ev.logMonoTime == 504887043297:
        # print(ev.initData.gitCommit)
            print(ev)

    # if ev.which() == "navInstruction":
    #     if ev.logMonoTime == 504887043297:
    #     # print(ev.initData.gitCommit)
    #         print(ev)

    # if ev.logMonoTime == 501376971354:
    #     print(ev)


# import pickle
# with open('/home/adas/openpilot/selfdrive/modeld/models/driving_policy_metadata.pkl', 'rb') as f:
#       model_metadata = pickle.load(f)
#       output_slices = model_metadata['output_slices']

#       for k,v in output_slices.items():
#            print(k,v)


# import pickle
# with open('/home/adas/FrogPilot/openpilot/frogpilot/classic_modeld/models/supercombo_metadata.pkl', 'rb') as f:
#       print('supercombo_metadata.pkl')
#       model_metadata = pickle.load(f)
#       output_slices = model_metadata['output_slices']

#       for k,v in output_slices.items():
#            print(k,v)