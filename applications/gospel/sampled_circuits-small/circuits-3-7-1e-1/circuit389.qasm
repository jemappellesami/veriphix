OPENQASM 2.0;
include "qelib1.inc";
qreg q390[3];
rx(pi/4) q390[2];
rz(7*pi/4) q390[2];
rx(3*pi/4) q390[2];
cx q390[1],q390[2];
cx q390[1],q390[0];
