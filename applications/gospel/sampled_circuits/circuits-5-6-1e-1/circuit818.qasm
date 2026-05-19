OPENQASM 2.0;
include "qelib1.inc";
qreg q819[5];
cx q819[3],q819[4];
cx q819[2],q819[3];
cx q819[1],q819[2];
cx q819[1],q819[0];
rx(pi/4) q819[1];
