import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

def precision_graphs_compare(question):
    color_dict = {'BernoulliNB': 'b', 'SVC': 'g', 'NuSVC': 'r', 'SGDClassifier': 'c'}
    question = question
    filename = "algorithmresultsquestion{0}new.log".format(question)
    outputname = "algorithmresults{0}compareplot.pdf".format(question)

    clfdata = defaultdict(list)
    with open(filename, 'r') as inputfile:
        lines = map(lambda x: x.strip(), inputfile.readlines())
        lines = map(lambda x: x.split('\t'), lines)
        lines = filter(lambda x: x[0] != 'Classifier', lines)

    for line in lines:
        clfdata[line[0], line[1]].append(line[2:])

    plot_data = []
    for key in clfdata.keys():
        tmp_data = []
        for i in clfdata[key]:
            tmp_data.append([key, i[0], i[1], i[2]])
        plot_data.append(tmp_data)

    fig = plt.figure(figsize=(16,10), dpi=300)
    sub1 = plt.subplot(111)
    for i in range(len(plot_data)):
        x = list(map(lambda x: float(x[2]), plot_data[i]))
        y = list(map(lambda x: x[1], plot_data[i]))
        c = Counter(zip(x,y))
        s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
        axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
        axisy = list(map(lambda x: x[1], plot_data[i]))
        sub1.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
        sub1.set_xlabel('precision')
        sub1.set_ylabel('training size')
    
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[3]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub2.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub2.set_xlabel('recall')
    #    sub2.set_ylabel('training size')

    box = sub1.get_position()
    sub1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub2.get_position()
    #sub2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub3.get_position()
    #sub3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub4.get_position()
    #sub4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    sub1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    #plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #plt.show()
    plt.savefig(outputname)

def precision_graphs(question):
    color_dict = {'BernoulliNB': 'b', 'SVC': 'g', 'NuSVC': 'r', 'SGDClassifier': 'c'}
    question = question
    filename = "algorithmresultsquestion{0}new.log".format(question)
    outputname = "algorithmresults{0}precisionplot.pdf".format(question)

    clfdata = defaultdict(list)
    with open(filename, 'r') as inputfile:
        lines = map(lambda x: x.strip(), inputfile.readlines())
        lines = map(lambda x: x.split('\t'), lines)
        lines = filter(lambda x: x[0] != 'Classifier', lines)

    for line in lines:
        clfdata[line[0], line[1]].append(line[2:])

    plot_data = []
    for key in clfdata.keys():
        tmp_data = []
        for i in clfdata[key]:
            tmp_data.append([key, i[0], i[1], i[2]])
        plot_data.append(tmp_data)

    fig = plt.figure(figsize=(16,10), dpi=300)
    sub1 = plt.subplot(221)
    sub2 = plt.subplot(222)
    sub3 = plt.subplot(223)
    sub4 = plt.subplot(224)
    for i in range(len(plot_data)):
        if plot_data[i][0][0][0] ==  "BernoulliNB":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub1.scatter(x, y, c=color_dict["BernoulliNB"], label=plot_data[i][0][0][0].split()[0], s=s)
            sub1.set_xlabel('precision')
            sub1.set_ylabel('training size')
            sub1.set_title('BernoulliNB')
            sub1.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "SVC":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub2.scatter(x, y, c=color_dict['SVC'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub2.set_xlabel('precision')
            sub2.set_ylabel('training size')
            sub2.set_title('SVC')
            sub2.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "SGDClassifier":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub3.scatter(x, y, c=color_dict['SGDClassifier'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub3.set_xlabel('precision')
            sub3.set_ylabel('training size')
            sub3.set_title('SGDClassifier')
            sub3.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "NuSVC":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub4.scatter(x, y, c=color_dict['NuSVC'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub4.set_xlabel('precision')
            sub4.set_ylabel('training size')
            sub4.set_title('NuSVC')
            sub4.set_xlim(0,1)
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[2]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub1.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub1.set_xlabel('precision')
    #    sub1.set_ylabel('training size')
    #
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[3]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub2.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub2.set_xlabel('recall')
    #    sub2.set_ylabel('training size')

    #box = sub1.get_position()
    #sub1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub2.get_position()
    #sub2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub3.get_position()
    #sub3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub4.get_position()
    #sub4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    #sub1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    #plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #plt.show()
    plt.savefig(outputname)

def error_graphs(question):
    color_dict = {'BernoulliNB': 'b', 'SVC': 'g', 'NuSVC': 'r', 'SGDClassifier': 'c'}
    question = question
    filename = "algorithmresultsquestion{0}new.log".format(question)
    outputname = "algorithmresults{0}errorplot.pdf".format(question)

    clfdata = defaultdict(list)
    with open(filename, 'r') as inputfile:
        lines = map(lambda x: x.strip(), inputfile.readlines())
        lines = map(lambda x: x.split('\t'), lines)
        lines = filter(lambda x: x[0] != 'Classifier', lines)

    for line in lines:
        clfdata[line[0], line[1]].append(line[2:])

    plot_data = []
    for key in clfdata.keys():
        tmp_data = []
        for i in clfdata[key]:
            tmp_data.append([key, i[0], i[4]])
        plot_data.append(tmp_data)

    fig = plt.figure(figsize=(16,10), dpi=300)
    sub1 = plt.subplot(221)
    sub2 = plt.subplot(222)
    sub3 = plt.subplot(223)
    sub4 = plt.subplot(224)
    for i in range(len(plot_data)):
        if plot_data[i][0][0][0] ==  "BernoulliNB":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub1.scatter(x, y, c=color_dict['BernoulliNB'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub1.set_xlabel('error')
            sub1.set_ylabel('training size')
            sub1.set_title('BernoulliNB')
            sub1.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "SVC":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub2.scatter(x, y, c=color_dict['SVC'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub2.set_xlabel('error')
            sub2.set_ylabel('training size')
            sub2.set_title('SVC')
            sub2.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "SGDClassifier":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub3.scatter(x, y, c=color_dict['SGDClassifier'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub3.set_xlabel('error')
            sub3.set_ylabel('training size')
            sub3.set_title('SGDClassifier')
            sub3.set_xlim(0,1)

    for i in range(len(plot_data)):
        if plot_data[i][0][0][0].split()[0] ==  "NuSVC":
            x = list(map(lambda x: float(x[2]), plot_data[i]))
            y = list(map(lambda x: x[1], plot_data[i]))
            c = Counter(zip(x,y))
            s = [10*c[(xx,yy)] for xx,yy in zip(x,y)]
            axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
            axisy = list(map(lambda x: x[1], plot_data[i]))
            sub4.scatter(x, y, c=color_dict['NuSVC'], label=plot_data[i][0][0][0].split()[0], s=s)
            sub4.set_xlabel('error')
            sub4.set_ylabel('training size')
            sub4.set_title('NuSVC')
            sub4.set_xlim(0,1)
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[2]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub1.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub1.set_xlabel('precision')
    #    sub1.set_ylabel('training size')
    #
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[3]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub2.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub2.set_xlabel('recall')
    #    sub2.set_ylabel('training size')

    #box = sub1.get_position()
    #sub1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub2.get_position()
    #sub2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub3.get_position()
    #sub3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub4.get_position()
    #sub4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    #sub1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    #plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #plt.show()
    plt.savefig(outputname)

def error_graphs_compare(question):
    color_dict = {'BernoulliNB': 'b', 'SVC': 'g', 'NuSVC': 'r', 'SGDClassifier': 'c'}
    question = question
    filename = "algorithmresultsquestion{0}new.log".format(question)
    outputname = "algorithmresults{0}errorcompareplot.pdf".format(question)

    clfdata = defaultdict(list)
    with open(filename, 'r') as inputfile:
        lines = map(lambda x: x.strip(), inputfile.readlines())
        lines = map(lambda x: x.split('\t'), lines)
        lines = filter(lambda x: x[0] != 'Classifier', lines)

    for line in lines:
        clfdata[line[0], line[1]].append(line[2:])

    plot_data = []
    for key in clfdata.keys():
        tmp_data = []
        for i in clfdata[key]:
            tmp_data.append([key, i[0], i[4]])
        plot_data.append(tmp_data)

    fig = plt.figure(figsize=(16,10), dpi=300)
    sub1 = plt.subplot(111)
    for i in range(len(plot_data)):
        x = list(map(lambda x: float(x[2]), plot_data[i]))
        y = list(map(lambda x: x[1], plot_data[i]))
        c = Counter(zip(x,y))
        s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
        axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
        axisy = list(map(lambda x: x[1], plot_data[i]))
        sub1.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
        sub1.set_xlabel('error')
        sub1.set_ylabel('training size')
    
    #for i in range(len(plot_data)):
    #    x = list(map(lambda x: float(x[3]), plot_data[i]))
    #    y = list(map(lambda x: x[1], plot_data[i]))
    #    c = Counter(zip(x,y))
    #    s = [100*c[(xx,yy)] for xx,yy in zip(x,y)]
    #    axisx = list(map(lambda x: int(10*float(x[2]))/10, plot_data[i]))
    #    axisy = list(map(lambda x: x[1], plot_data[i]))
    #    sub2.scatter(x, y, c=color_dict[i], label=plot_data[i][0][0][0].split()[0], s=s)
    #    sub2.set_xlabel('recall')
    #    sub2.set_ylabel('training size')

    box = sub1.get_position()
    sub1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub2.get_position()
    #sub2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub3.get_position()
    #sub3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    #box = sub4.get_position()
    #sub4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    sub1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #sub4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    #plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #plt.show()
    plt.savefig(outputname)
error_graphs(0)
error_graphs(1)
error_graphs(2)
precision_graphs(0)
precision_graphs(1)
precision_graphs(2)
#precision_graphs_compare(0)
#precision_graphs_compare(1)
#precision_graphs_compare(2)
#error_graphs_compare(0)
#error_graphs_compare(1)
#error_graphs_compare(2)
