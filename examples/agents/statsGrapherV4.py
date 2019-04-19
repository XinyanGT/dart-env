__author__ = 'alexander_clegg'
import numpy as np
import os
import math

import pyPhysX.pyutils as pyutils
import pyPhysX.renderUtils as renderutils

#Note: this version allows:
#1. preferred min/max over a matrix of graphs
#2. option: all-graph or mean/variance
#3. option: combo/overlay or matrix

if __name__ == '__main__':

    #1. set variables
    compute_success_percent = True
    success_threshold = 0.8
    filemode = False
    legend = False
    graphStats = False #if true, graph mean/variance instead of data
    singleFrame = False #if true, graph everything on the same graph
    graph0 = True #if true, put a black line through 0 y

    x_axis_title = "Timestep"
    y_axis_title = "Progress"
    #y_axis_title = "Force (Newtons)"

    ymax = None
    ymin = None
    #ymax = 100.0
    #ymax = 200

    #limb progress
    ymax = 1.0
    ymin = -2.0

    #forces
    #ymax = 700.0
    #ymin = 0

    unifyScale = True #if true and no limits provided, compute from data min and max
    #graphTitle = "Limb Progress"
    graphTitle = "Forces"

    #2. set closest common directory
    #prefix = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/"
    #prefix = "/home/alexander/Documents/dev/"
    prefix = "/home/alexander/Documents/dev/data_recording_dir/curr_expandedtanh/"
    prefixes = [prefix]

    #typical trials

    prefixes_capable_onearm = [
        "/home/alexander/Documents/dev/data_recording_dir/typical/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_linear/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_typical_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_slowhr_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/fresh_typical_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_typical_nocap/"

    ]

    prefixes_tremor_onearm = [
        "/home/alexander/Documents/dev/data_recording_dir/tremor/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_tremor_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_tremor_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_tremor_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_tremor_nocap/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_tremor_linearpenx10_low/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_tremor_linearpenx10_moderate/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_tremor_linearpenx10_high/"
    ]

    prefixes_weakness_onearm = [
        "/home/alexander/Documents/dev/data_recording_dir/weakness/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_weak_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_weak_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_weakstrong_nocap/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_weak/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_moderate/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_strong/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_weakstrong_nohumanjobs/"
    ]

    prefixes_weakness_onearm_variations = [
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_weak/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_moderate/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10_strong/"

    ]

    prefixes_jcon_onearm = [
        "/home/alexander/Documents/dev/data_recording_dir/elbow_constraint/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_elbowcon_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_elbowconstraint_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_jcon_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_jcon_nocap/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_jcon_nohumanjobs/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_jcon_nohumanobs/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_jcon_linearpenx10_low/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_jcon_linearpenx10_middle/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_jcon_linearpenx10_high/"

    ]

    prefixes_typical_training = [
        "/home/alexander/Documents/dev/data_recording_dir/elbow_constraint/",
        "/home/alexander/Documents/dev/data_recording_dir/tremor/",
        "/home/alexander/Documents/dev/data_recording_dir/weakness/",
        "/home/alexander/Documents/dev/data_recording_dir/typical/"
        "/home/alexander/Documents/dev/data_recording_dir/vel025/"
    ]


    #curriculum trials
    prefixes_curr = [
        "/home/alexander/Documents/dev/data_recording_dir/curr_expandedtanh/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_+tanh/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_linear/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_typical_shortH/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_weak_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_tremor_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_slowrobo_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/curr_elbowcon_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_typical_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_tremor_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_jcon_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_curr_weakstrong_linearpenx10/"

    ]

    #fresh linear contact pen trials

    prefixes_fresh = [
        "/home/alexander/Documents/dev/data_recording_dir/fresh_typical_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_tremor_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_weak_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_fresh_elbowconstraint_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_slowrobo_fresh_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/onearm_slowhr_linearpen/"
    ]


    #two arm trials

    prefixes_twoarm = [
        "/home/alexander/Documents/dev/data_recording_dir/curr_twoarm_linearconpen/",
        "/home/alexander/Documents/dev/data_recording_dir/twoarm_gown/",
        "/home/alexander/Documents/dev/data_recording_dir/twoarmgown_fresh_linearpen/",
        "/home/alexander/Documents/dev/data_recording_dir/twoarm_gown_curr_linearpenx10/",
        "/home/alexander/Documents/dev/data_recording_dir/twoarm_gown_nocap/"
    ]

    prefixes_raw100 = [
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/jcon/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/jcon_x10/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/tremor/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/tremor_x10/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/typical/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/typical_x10/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_x10/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/twoarm_gown/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/twoarm_gown_x10/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_nocap/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_nohumanjobs/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/jcon_x10_low/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/jcon_x10_middle/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/jcon_x10_high/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_x10_weak/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_x10_moderate/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/weakstrong_x10_strong/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/typical_on_tremor/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/typical_on_weakness/",
        "/home/alexander/Documents/dev/data_recording_dir/100x_raw_data/typical_on_jcon/"
    ]

    prefix_list = [
        prefixes_capable_onearm,
        prefixes_tremor_onearm,
        prefixes_weakness_onearm,
        prefixes_jcon_onearm
    ]

    prefix_list = [
        prefixes_raw100
    ]

    success_percents = []
    for prefixes in prefix_list:

        for prix,prefix in enumerate(prefixes):
            #define the matrix structure with remaining directory info:
            #folders = [
            #    ["1", "2"],
            #    ["3", "4"]
            #           ]

            #elbow variation
            folders = [
                ["0", "1", "2"],
                ["3", "4", "5"],
                ["6", "7", "8"]
            ]
            titles = None
            titles = [
                ["0", "0.125", "0.25"],
                ["0.375", "0.5", "0.625"],
                ["0.75", "0.875", "1.0"]
            ]
            #folders = [['baseline']]

            folders = [[""]]
            titles = [["Limb Progress"]]

            #filename = "limbProgressGraphData"
            #filename = "deformationGraphData"
            filename = "progress_history0.txt"



            #filenames = [[ "progress_history0.txt", "progress_history1.txt"]]

            #titles = [[ "limb 0 progress", "limb 1 progress"]]

            '''
            filenames = [
                ["max_cloth_contact", "total_cloth_contact"],
                ["max_rigid_contact", "total_rigid_contact"],
                ["max_contact", "total_contact"]
            ]
    
            titles = [
                ["Maximum Cloth Contact Force", "total_cloth_contact"],
                ["max_rigid_contact", "total_rigid_contact"],
                ["max_contact", "total_contact"]
            ]

            '''

            '''
            filenames = [
                ["max_cloth_contact"]
                #["max_rigid_contact"]
            ]

            titles = [
                ["Maximum Cloth Contact Force"]
                #["Maximum Rigid Contact Force"]
            ]

            '''

            if graphStats:
                for lix,list in enumerate(titles):
                    for tix,title in enumerate(list):
                        list[tix] = title + "_stats"

            print(titles)

            inprefixs = []

            if filemode:
                #file based method
                for f_row in filenames:
                    inprefixs.append([])
                    for f in f_row:
                        inprefixs[-1].append(prefix + f)
            else:
                #folder based method

                for f_row in folders:
                    inprefixs.append([])
                    for f in f_row:
                        inprefixs[-1].append(prefix+f+"/")

            outprefix = prefix

            print("loading data")

            #labels = ["Linear", "RL Policy"]
            labels = ["label"]

            data = []
            for p_row in inprefixs:
                data.append([])
                for p in p_row:
                    if filemode:
                        print(p)
                        data[-1].append(pyutils.loadData2D(filename=p))
                    else:
                        print(p + filename)
                        data[-1].append(pyutils.loadData2D(filename=p+filename))

            #average over each timestep
            avgs = []
            vars = []

            print("data string length: " + str(len(data[0][0][0])))

            #compute the min and max y values if needed
            if (ymax is None or ymin is None) and unifyScale:
                print("computing min/max values:")
                maxy = -99999
                miny = 99999
                for r in range(len(data)):
                    for c in range(len(data[r])):
                        for s in range(len(data[r][c])):
                            for t in range(len(data[r][c][s])):
                                if data[r][c][s][t] < miny:
                                    miny = data[r][c][s][t]
                                if data[r][c][s][t] > maxy:
                                    maxy = data[r][c][s][t]
                print(" max: " + str(maxy) + ", min: " + str(miny))
                if(ymax is None):
                    ymax = maxy
                if(ymin is None):
                    ymin = miny

            if compute_success_percent:
                for r in range(len(data)):
                    for c in range(len(data[r])):
                        success_percent = 0
                        for s in range(len(data[r][c])):
                            is_success = False
                            for t in range(len(data[r][c][s])):
                                if data[r][c][s][t] >= success_threshold:
                                    is_success = True
                            if is_success:
                                success_percent += 1
                        success_percent /= len(data[r][c])
                        success_percents.append(success_percent)

            if graphStats:
                #compute averages
                for r in range(len(data)):
                    avgs.append([])
                    #compute averages of these lists
                    for c in range(len(data[r])):
                        avgs[-1].append([])

                        count = []
                        for s in range(len(data[r][c])):
                            for t in range(len(data[r][c][s])):
                                if(t > len(count)-1):
                                    count.append(0)
                                    avgs[-1][-1].append(0)
                                avgs[-1][-1][t] += data[r][c][s][t]
                                count[t] += 1
                        for t in range(len(avgs[-1][-1])):
                            avgs[-1][-1][t] /= count[t]

                #compute variances
                for r in range(len(data)):
                    vars.append([])
                    #compute averages of these lists
                    for c in range(len(data[r])):
                        vars[-1].append([])

                        count = []
                        for s in range(len(data[r][c])):
                            for t in range(len(data[r][c][s])):
                                if(t > len(count)-1):
                                    count.append(0)
                                    vars[-1][-1].append(0)
                                vars[-1][-1][t] += (data[r][c][s][t] - avgs[r][c][t])*(data[r][c][s][t] - avgs[r][c][t])
                                count[t] += 1
                        for t in range(len(vars[-1][-1])):
                            vars[-1][-1][t] /= count[t]
                            #std dev
                            vars[-1][-1][t] = math.sqrt(vars[-1][-1][t])

            #if compressing to 1 frame, re-organize the data into one group
            if singleFrame:
                newdata = []

                xdim = 0

                if graphStats:
                    # add average curve and 2 variance curves per entry
                    for r in avgs:
                        for c in r:
                            newdata.append(c)
                    for rix,r in enumerate(vars):
                        for cix,c in enumerate(r):
                            newdata.append([])
                            newdata.append([])
                            if len(c) > xdim:
                                xdim = len(c)
                            for tix,t in enumerate(c):
                                newdata[-1].append(avgs[rix][cix][tix] + t)
                                newdata[-2].append(avgs[rix][cix][tix] - t)
                else:
                    #re-group all curves into one graph
                    for r in data:
                        for c in r:
                            for s in c:
                                newdata.append(s)
                                if len(s) > xdim:
                                    xdim = len(s)

                graph = None

                if unifyScale or ymax is not None or ymax is not None:
                    graph = pyutils.LineGrapher(title=graphTitle, legend=legend, ylims=(ymin, ymax), x_axis_title=x_axis_title, y_axis_title=y_axis_title)
                else:
                    graph = pyutils.LineGrapher(title=graphTitle, legend=legend, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

                graph.xdata = np.arange(xdim)

                for dix,d in enumerate(newdata):
                    if graphStats:
                        graph.plotData(ydata=d)
                        if(dix > len(newdata)/3): #its a variance, so recolor to mean
                            avg_ix = int((dix-len(newdata)/3)/2)
                            pc = graph.getPlotColor(avg_ix)
                            #spc = str(pc).lstrip('#')
                            print("plot color: " + str(pc))
                            #rgb = tuple(int(spc[i:i + 2], 16) for i in (0, 2, 4))
                            #print('RGB =', rgb)
                            #nrgb = (min(256, int(rgb[0]*1.2)), min(256, int(rgb[1]*1.2)), min(256, int(rgb[2]*1.2)))
                            #print('nrgb =', nrgb)
                            #nhex = '#%02x%02x%02x' % nrgb
                            #print('nhex =', nhex)
                            #newcolor = graph.lighten_color(pc)
                            #if graph0:
                            #    graph.plotData(ydata=np.zeros(xdim), color=[0, 0, 0])
                            ##graph.plotData(ydata=d, color=graph.colors[avg_ix]*1.2) #make it lighter
                            #graph.plotData(ydata=d, color=newcolor) #make it lighter
                            #TODO: fix this
                    else:
                        #graph.plotData(ydata=d)
                        graph.yData.append(d)

                if not graphStats:
                    graph.update()

                graph.save(filename=outprefix+graphTitle)

            else:
                xdim = 0
                infilenames = []

                #first create individual graphs and save them
                if graphStats:
                    newdata = []
                    for rix,r in enumerate(avgs):
                        newdata.append([])
                        infilenames.append([])
                        for cix,c in enumerate(r):
                            infilenames[-1].append(outprefix+"g_"+str(rix)+"_"+str(cix)+".png")
                            newdata[-1].append([])
                            newdata[-1][-1].append(c)
                            if len(c) > xdim:
                                xdim = len(c)
                            newdata[-1][-1].append([])
                            newdata[-1][-1].append([])
                            for tix,t in enumerate(c):
                                newdata[rix][cix][-1].append(c[tix] + vars[rix][cix][tix])
                                newdata[rix][cix][-2].append(c[tix] - vars[rix][cix][tix])

                            graph = None
                            if(titles is not None):
                                graphTitle = titles[rix][cix]
                                if compute_success_percent:
                                    graphTitle = graphTitle + "\n" + str(int(success_percents[prix]*100)) +"% successful"
                            if unifyScale or ymax is not None or ymax is not None:
                                graph = pyutils.LineGrapher(title=graphTitle, legend=legend, ylims=(ymin, ymax), x_axis_title=x_axis_title, y_axis_title=y_axis_title)
                            else:
                                graph = pyutils.LineGrapher(title=graphTitle, legend=legend, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

                            graph.xdata = np.arange(xdim)

                            if graph0:
                                graph.plotData(ydata=np.zeros(xdim), color=[0, 0, 0])

                            graph.plotData(ydata=newdata[rix][cix][1], color=[0.6, 0.6, 1.0])
                            graph.plotData(ydata=newdata[rix][cix][2], color=[0.6, 0.6, 1.0])

                            graph.plotData(ydata=newdata[rix][cix][0], color=[0,0,1.0])


                            #save the graphs
                            graph.save(filename=outprefix+"g_"+str(rix)+"_"+str(cix))
                else:
                    #simply re-graph the data with potentially unified scale

                    for rix,r in enumerate(data):
                        infilenames.append([])
                        for cix,c in enumerate(r):
                            infilenames[-1].append(outprefix + "g_" + str(rix) + "_" + str(cix)+".png")
                            for s in c:
                                if len(s) > xdim:
                                    xdim = len(s)

                    for rix,r in enumerate(data):
                        for cix,c in enumerate(r):
                            graph = None
                            if (titles is not None):
                                graphTitle = titles[rix][cix]
                                if compute_success_percent:
                                    graphTitle = graphTitle + "\n" + str(int(success_percents[prix] * 100)) + "% successful"

                            if unifyScale or ymax is not None or ymax is not None:
                                graph = pyutils.LineGrapher(title=graphTitle, legend=legend, ylims=(ymin, ymax), x_axis_title=x_axis_title, y_axis_title=y_axis_title)
                            else:
                                graph = pyutils.LineGrapher(title=graphTitle, legend=legend, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

                            graph.xdata = np.arange(xdim)

                            if graph0:
                                graph.plotData(ydata=np.zeros(xdim), color=[0, 0, 0])

                            for six,s in enumerate(c):
                                graph.plotData(ydata=data[rix][cix][six], update=False)
                            graph.update()

                            #save the graphs
                            graph.save(filename=outprefix+"g_" + str(rix) + "_" + str(cix))

                print("infilenames: " + str(infilenames))
                #then create an image matrix for the graphs
                renderutils.imageMatrixFrom(filenames=infilenames, outfilename=outprefix+graphTitle)

                if compute_success_percent:
                    print("Success Percents:")
                    print(success_percents)
                    #avg_Graph.save(filename=outprefix+"progress_avg_Graph")
