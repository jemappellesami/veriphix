OPENQASM 2.0;
include "qelib1.inc";
qreg q775[4];
rx(3*pi/4) q775[3];
cx q775[2],q775[3];
cx q775[1],q775[2];
cx q775[0],q775[1];
