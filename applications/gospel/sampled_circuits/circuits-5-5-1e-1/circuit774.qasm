OPENQASM 2.0;
include "qelib1.inc";
qreg q775[5];
cx q775[3],q775[4];
cx q775[2],q775[3];
cx q775[2],q775[1];
cx q775[0],q775[1];
