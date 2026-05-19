OPENQASM 2.0;
include "qelib1.inc";
qreg q340[5];
cx q340[3],q340[4];
cx q340[3],q340[2];
cx q340[1],q340[2];
cx q340[1],q340[0];
rx(pi/4) q340[1];
