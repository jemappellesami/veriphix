OPENQASM 2.0;
include "qelib1.inc";
qreg q334[3];
rx(5*pi/4) q334[2];
rz(3*pi/2) q334[2];
cx q334[2],q334[1];
cx q334[0],q334[1];
