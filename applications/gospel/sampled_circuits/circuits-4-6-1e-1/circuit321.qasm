OPENQASM 2.0;
include "qelib1.inc";
qreg q322[4];
cx q322[3],q322[2];
rx(3*pi/4) q322[2];
cx q322[2],q322[1];
cx q322[0],q322[1];
rx(pi/4) q322[1];
