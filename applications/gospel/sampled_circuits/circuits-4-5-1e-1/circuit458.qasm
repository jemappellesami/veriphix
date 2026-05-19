OPENQASM 2.0;
include "qelib1.inc";
qreg q459[4];
rx(pi/4) q459[3];
cx q459[2],q459[3];
cx q459[2],q459[1];
cx q459[0],q459[1];
