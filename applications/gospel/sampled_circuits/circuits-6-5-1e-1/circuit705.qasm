OPENQASM 2.0;
include "qelib1.inc";
qreg q706[6];
cx q706[4],q706[5];
cx q706[4],q706[3];
cx q706[2],q706[3];
cx q706[2],q706[1];
cx q706[0],q706[1];
