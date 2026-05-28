OPENQASM 2.0;
include "qelib1.inc";
qreg q803[3];
cx q803[1],q803[0];
rz(3*pi/4) q803[2];
rx(3*pi/2) q803[0];
cx q803[2],q803[1];
cx q803[1],q803[0];
