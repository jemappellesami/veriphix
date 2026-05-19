OPENQASM 2.0;
include "qelib1.inc";
qreg q58[3];
rz(pi/4) q58[1];
rx(5*pi/4) q58[2];
cx q58[1],q58[2];
cx q58[1],q58[0];
rx(pi/4) q58[1];
