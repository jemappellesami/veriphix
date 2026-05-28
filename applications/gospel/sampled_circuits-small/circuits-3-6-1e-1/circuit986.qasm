OPENQASM 2.0;
include "qelib1.inc";
qreg q987[3];
rx(3*pi/4) q987[2];
cx q987[2],q987[1];
cx q987[1],q987[0];
rx(pi/4) q987[1];
