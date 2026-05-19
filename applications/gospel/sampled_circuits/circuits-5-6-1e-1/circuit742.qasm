OPENQASM 2.0;
include "qelib1.inc";
qreg q743[5];
cx q743[4],q743[3];
cx q743[3],q743[2];
cx q743[2],q743[1];
cx q743[0],q743[1];
rx(pi/4) q743[1];
