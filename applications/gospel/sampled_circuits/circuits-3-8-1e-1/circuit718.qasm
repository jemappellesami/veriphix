OPENQASM 2.0;
include "qelib1.inc";
qreg q719[3];
rz(pi/2) q719[2];
cx q719[2],q719[1];
rx(3*pi/2) q719[1];
cx q719[1],q719[0];
rx(pi/4) q719[1];
