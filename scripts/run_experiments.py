#! /usr/bin/python
import os
import yaml
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run placement experiments specified by a list of yaml files")
    parser.add_argument('yaml_template', type=str, help='Path to a folder containing problem definition templates')
    # parser.add_argument('options', type=str, help='Path to a yaml file containing parameter options')
    parser.add_argument('num_runs', type=int, help='Number of runs for each problem')
    args = parser.parse_args()
    test_script_path = os.path.abspath(os.path.dirname(__file__)) + "/test_scripts/test_placement2.py"
    # first read options file
    # with open(args.options, 'r') as options_file:
    #     options = yaml.load(options_file)
    weights = [(0.1, 0.3, 0.3, 0.3), (0.2, 0.2, 0.4, 0.2), (0.25, 0.25, 0.25, 0.25)]
    relax_types = ['sub-binary', 'continuous']
    bodies = ['crayola']
    cs = [0.5, 0.8, 1.0]
    # read template
    with open(args.yaml_template) as template_file:
        yaml_template = template_file.read()
    yaml_folder = os.path.abspath(os.path.dirname(args.yaml_template))
    yaml_instance_filename = yaml_folder + '/yaml_instance.yaml'
    command_list = ["python", test_script_path, yaml_instance_filename, '--num_runs', str(args.num_runs)]
    # run experiments for binary
    for body in bodies:
        for c in cs:
            yaml_instance = yaml_template.replace('<BODY_NAME>', body)
            yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'binary')
            yaml_instance = yaml_instance.replace('<ARM_WEIGHT>', '0.1')
            yaml_instance = yaml_instance.replace('<OBJ_COL_WEIGHT>', '0.3')
            yaml_instance = yaml_instance.replace('<CONTACT_WEIGHT>', '0.3')
            yaml_instance = yaml_instance.replace('<OBJ_WEIGHT>', '0.3')
            yaml_instance = yaml_instance.replace('<C>', str(c))
            exp_id = 'binary_' + str(c)
            yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
            # dump yaml into same folder as template (because of grasps etc)
            with open(yaml_instance_filename, 'w') as tmp_yaml_file:
                tmp_yaml_file.write(yaml_instance)
            # print "Fake call %s" % str(command_list)
            subprocess.call(command_list)
    # run experiments for other relaxation types
    for body in bodies:
        for relax in relax_types:
            for ws in weights:
                for c in cs:
                    yaml_instance = yaml_template.replace('<BODY_NAME>', body)
                    yaml_instance = yaml_instance.replace('<RELAX_TYPE>', relax)
                    yaml_instance = yaml_instance.replace('<ARM_WEIGHT>', str(ws[0]))
                    yaml_instance = yaml_instance.replace('<OBJ_COL_WEIGHT>', str(ws[1]))
                    yaml_instance = yaml_instance.replace('<CONTACT_WEIGHT>', str(ws[2]))
                    yaml_instance = yaml_instance.replace('<OBJ_WEIGHT>', str(ws[3]))
                    yaml_instance = yaml_instance.replace('<C>', str(c))
                    exp_id = relax + '_' + str(c)
                    yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                    # dump yaml into same folder as template (because of grasps etc)
                    with open(yaml_instance_filename, 'w') as tmp_yaml_file:
                        tmp_yaml_file.write(yaml_instance)
                    # print "Fake call %s" % str(command_list)
                    subprocess.call(command_list)
