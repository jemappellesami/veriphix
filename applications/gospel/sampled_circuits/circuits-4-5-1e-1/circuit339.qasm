OPENQASM 2.0;
include "qelib1.inc";
qreg q340[4];
rx(7*pi/4) q340[0];
cx q340[0],q340[1];
cx q340[1],q340[2];
rx(pi/4) q340[0];
