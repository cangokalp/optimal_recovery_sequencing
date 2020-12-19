for rep in [3]:
    for num_broken in ['16']:
        SCENARIO_DIR = NETWORK_DIR
        ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))
        ULT_SCENARIO_REP_DIR = os.path.join(
            ULT_SCENARIO_DIR, str(rep))
        save_dir = ULT_SCENARIO_REP_DIR

        start = time.time()
        net_before = create_network(NETFILE, TRIPFILE)
        net_before.not_fixed = set([])
        before_eq_tstt = solve_UE(net=net_before, eval_seq=True, flows=True)
        add_if_time = time.time() - start

        ddict = load(save_dir + '/' + 'damaged_dict')
        damaged_links = ddict.keys()
        start = time.time()
        net_after = create_network(NETFILE, TRIPFILE)
        net_after.not_fixed = set(damaged_links)
        after_eq_tstt = solve_UE(net=net_after, eval_seq=True)
        add_net_after_time = time.time() - start
        bb = {}
        links_to_remove = deepcopy(list(damaged_links))
        to_visit = links_to_remove
        added = []
        for link in links_to_remove:
            test_net = deepcopy(net_after)
            added = [link]
            not_fixed = set(to_visit).difference(set(added))
            test_net.not_fixed = set(not_fixed)
            tstt_after = solve_UE(net=test_net)
            bb[link] = max(after_eq_tstt - tstt_after, 0)

        ordered_days = []
        orderedb_benefits = []

        sorted_d = sorted(ddict.items(), key=lambda x: x[1])
        for key, value in sorted_d:
            ordered_days.append(value)
            orderedb_benefits.append(bb[key])

        ob, od, lzg_order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)
        lzg_order = [i[0] for i in lzg_order]
        bb_time = time.time() - start + add_if_time

        bound, eval_taps, _ = eval_sequence(net_before, lzg_order, after_eq_tstt, before_eq_tstt, damaged_dict=ddict)
        tap_solved = len(ddict)

        fname = save_dir + '/layzgreedy_solution'
        save(fname + '_obj', bound)
        save(fname + '_path', lzg_order)
        save(fname + '_elapsed', bb_time)
        save(fname + '_num_tap', tap_solved)
        save(save_dir + '/add_net_after_time', add_net_after_time)
        save(save_dir + '/if_forgotten_time', add_if_time)

get_tables(NETWORK_DIR)