OPENQASM 2.0;
include "qelib1.inc";
qreg q969[5];
rx(7*pi/4) q969[4];
cx q969[3],q969[4];
cx q969[2],q969[3];
cx q969[2],q969[1];
cx q969[1],q969[0];
